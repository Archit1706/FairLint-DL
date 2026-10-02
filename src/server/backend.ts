import * as vscode from 'vscode';
import { spawn, spawnSync, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import axios from 'axios';
import { getConfig } from '../config/settings';
import { updateStatusBar, resetStatusBar, setStatusBarError } from './statusBar';
import { showError } from './errors';

let serverProcess: ChildProcess | null = null;

/** Modules the backend cannot start without. */
const REQUIRED_MODULES = ['fastapi', 'uvicorn', 'torch', 'pandas', 'sklearn', 'shap', 'lime'];

export const PIP_INSTALL_HINT =
    'pip install torch fastapi uvicorn "pandas<3.0" "numpy<2.3" scikit-learn scipy lime shap';

/**
 * Report which of REQUIRED_MODULES a candidate interpreter is missing.
 * Returns null if the executable cannot be run at all.
 *
 * This check exists because an interpreter that merely *exists* is not enough.
 * A machine can easily have several Pythons (a Store build on PATH as `python3`,
 * a python.org build holding the packages), and picking the wrong one used to
 * fail silently: uvicorn never started, and the resulting connection failure was
 * reported to the user as an unrelated "empty CSV" error.
 */
function missingModules(pythonPath: string): string[] | null {
    // `import importlib.util` explicitly: a bare `import importlib` does not
    // reliably bind the `util` submodule, which made this probe raise on some
    // interpreters and look like "could not be executed".
    const probe = `import importlib.util;print(','.join(m for m in ${JSON.stringify(
        REQUIRED_MODULES,
    )} if importlib.util.find_spec(m) is None))`;
    const res = spawnSync(pythonPath, ['-c', probe], { encoding: 'utf8', timeout: 60000 });
    if (res.error || res.status !== 0) {
        return null;
    }
    return res.stdout.trim() ? res.stdout.trim().split(',') : [];
}

export interface PythonResolution {
    pythonPath: string | null;
    /** Every candidate tried, with why it was rejected. For the error message. */
    tried: { path: string; reason: string }[];
}

/**
 * Resolve a Python executable that can actually run the backend.
 *
 * Candidates are tried in order of how likely they are to be the one the user
 * provisioned, and each is verified to import the required modules before use.
 */
export function resolvePython(backendPath: string): PythonResolution {
    const candidates: string[] = [];
    const push = (p: string | undefined | null) => {
        if (p && !candidates.includes(p)) {
            candidates.push(p);
        }
    };

    // 1. A venv alongside the backend is unambiguous: it was made for this.
    const venvWin = path.join(backendPath, 'venv', 'Scripts', 'python.exe');
    const venvUnix = path.join(backendPath, 'venv', 'bin', 'python');
    if (fs.existsSync(venvWin)) {
        push(venvWin);
    }
    if (fs.existsSync(venvUnix)) {
        push(venvUnix);
    }

    // 2. Whatever the user has told VS Code to use.
    push(vscode.workspace.getConfiguration('python').get<string>('defaultInterpreterPath'));
    push(vscode.workspace.getConfiguration('fairlint-dl').get<string>('python.interpreterPath'));

    // 3. Plain interpreters on PATH. `python` first: on Windows `python3` is
    //    often the Microsoft Store build, which rarely has the packages.
    push('python');
    push('python3');

    const tried: { path: string; reason: string }[] = [];
    for (const candidate of candidates) {
        const missing = missingModules(candidate);
        if (missing === null) {
            tried.push({ path: candidate, reason: 'could not be executed' });
            continue;
        }
        if (missing.length > 0) {
            tried.push({ path: candidate, reason: `missing ${missing.join(', ')}` });
            continue;
        }
        console.log(`Using Python: ${candidate}`);
        return { pythonPath: candidate, tried };
    }

    return { pythonPath: null, tried };
}

export async function startBackend(context: vscode.ExtensionContext): Promise<void> {
    const config = getConfig();
    const serverPort = config.serverPort;
    const serverUrl = `http://localhost:${serverPort}`;

    // Check if server is already running
    try {
        await axios.get(`${serverUrl}/`, { timeout: 1000 });
        console.log(`Server already running at ${serverUrl}`);
        updateStatusBar('$(check) Server Connected', 'Connected to existing Python backend');
        vscode.window.showInformationMessage('Connected to existing Fairness Analysis Server!');
        return;
    } catch {
        console.log(`Server not running, starting new instance...`);
    }

    const backendPath = context.asAbsolutePath('python_backend');

    // Check that python_backend directory exists
    if (!fs.existsSync(backendPath)) {
        setStatusBarError('Backend not found');
        showError(
            'Backend Not Found',
            `The python_backend directory was not found at:\n${backendPath}\n\n` +
                'This may indicate a packaging issue. Please reinstall the extension.',
        );
        return;
    }

    // Pick an interpreter that can actually import the backend's dependencies.
    // Reported up front, because a half-usable interpreter otherwise fails later
    // as an unrelated-looking connection error.
    updateStatusBar('$(sync~spin) Finding Python...', 'Looking for an interpreter with the backend dependencies');
    const { pythonPath, tried } = resolvePython(backendPath);

    if (!pythonPath) {
        setStatusBarError('No usable Python');
        const detail = tried.length
            ? tried.map((t) => `  • ${t.path} — ${t.reason}`).join('\n')
            : '  (no interpreters found)';
        showError(
            'Python Dependencies Missing',
            `No Python interpreter with the required packages was found.\n\n` +
                `Interpreters tried:\n${detail}\n\n` +
                `Install the dependencies with:\n  ${PIP_INSTALL_HINT}\n\n` +
                `If the packages are installed under a different interpreter, point VS Code at it ` +
                `via the "python.defaultInterpreterPath" setting.`,
        );
        return;
    }

    console.log(`Starting backend at: ${backendPath} on port ${serverPort}`);
    console.log(`Using Python: ${pythonPath}`);

    // Check requirements.txt exists and suggest install
    const requirementsPath = path.join(backendPath, 'requirements.txt');
    if (fs.existsSync(requirementsPath)) {
        console.log(`Requirements file found at: ${requirementsPath}`);
    }

    updateStatusBar('$(sync~spin) Starting server...', 'Initializing Python backend');

    serverProcess = spawn(pythonPath, ['-m', 'uvicorn', 'bias_server:app', '--port', String(serverPort)], {
        cwd: backendPath,
        shell: true,
    });

    serverProcess.stdout?.on('data', (data) => {
        console.log(`Backend stdout: ${data}`);
    });

    serverProcess.stderr?.on('data', (data) => {
        const msg = data.toString();
        // Uvicorn logs to stderr by default, so not all stderr is errors
        if (msg.includes('ModuleNotFoundError')) {
            console.error(`Backend ERROR: ${msg}`);
            const match = msg.match(/No module named '([^']+)'/);
            const moduleName = match ? match[1] : 'unknown';
            vscode.window
                .showErrorMessage(
                    `Missing Python module: ${moduleName}. Install backend dependencies?`,
                    'Install Dependencies',
                )
                .then((choice) => {
                    if (choice === 'Install Dependencies') {
                        const terminal = vscode.window.createTerminal('FairLint-DL Setup');
                        terminal.show();
                        terminal.sendText(`pip install -r "${path.join(backendPath, 'requirements.txt')}"`);
                    }
                });
        } else if (msg.includes('ERROR') || msg.includes('Traceback')) {
            console.error(`Backend ERROR: ${msg}`);
        } else {
            console.log(`Backend: ${msg}`);
        }
    });

    serverProcess.on('error', (err) => {
        console.error(`Failed to spawn backend process: ${err.message}`);
        setStatusBarError('Server spawn failed');
        showError(
            'Server Spawn Failed',
            `Could not start the Python backend process.\n` + `Python path: ${pythonPath}\n` + `Error: ${err.message}`,
        );
    });

    serverProcess.on('exit', (code, signal) => {
        if (code !== null && code !== 0) {
            console.error(`Backend process exited with code ${code}`);
            setStatusBarError('Server crashed');
            showError(
                'Server Exited Unexpectedly',
                `The backend server exited with code ${code}.\n` +
                    `Check the Debug Console for details.\n` +
                    `Python path used: ${pythonPath}`,
            );
        } else if (signal) {
            console.log(`Backend process killed by signal ${signal}`);
        }
    });

    await vscode.window.withProgress(
        {
            location: vscode.ProgressLocation.Notification,
            title: 'Starting Fairness Analysis Server...',
            cancellable: false,
        },
        async (progress) => {
            progress.report({ message: 'Initializing Python backend...' });

            let attempts = 0;
            const maxAttempts = 15;

            while (attempts < maxAttempts) {
                await new Promise((resolve) => setTimeout(resolve, 1000));
                attempts++;

                progress.report({
                    message: `Connecting to server (attempt ${attempts}/${maxAttempts})...`,
                    increment: 100 / maxAttempts,
                });
                updateStatusBar(
                    `$(sync~spin) Connecting... [${attempts}/${maxAttempts}]`,
                    'Waiting for backend server',
                );

                try {
                    await axios.get(`${serverUrl}/`, { timeout: 2000 });
                    vscode.window.showInformationMessage('Fairness Analysis Server is ready!');
                    resetStatusBar();
                    return;
                } catch {
                    // Continue trying
                }
            }

            setStatusBarError('Server failed');

            const requirementsFile = path.join(backendPath, 'requirements.txt');
            const installAction = 'Install Dependencies';
            const retryAction = 'Retry';
            const choice = await vscode.window.showErrorMessage(
                'Server Startup Failed: The backend server failed to start. Please check:\n' +
                    '1. Python 3.8+ is installed\n' +
                    '2. Required packages are installed\n' +
                    `3. Port ${serverPort} is not in use`,
                installAction,
                retryAction,
            );

            if (choice === installAction) {
                const terminal = vscode.window.createTerminal('FairLint-DL Setup');
                terminal.show();
                terminal.sendText(`pip install -r "${requirementsFile}"`);
            } else if (choice === retryAction) {
                stopBackend();
                await startBackend(context);
            }
        },
    );
}

export function stopBackend(): void {
    if (serverProcess && serverProcess.pid) {
        console.log(`Stopping backend server (PID: ${serverProcess.pid})...`);

        try {
            if (process.platform === 'win32') {
                // On Windows with shell: true, serverProcess is cmd.exe.
                // process.kill() only kills the shell, leaving Python orphaned.
                // taskkill /T kills the entire process tree.
                spawn('taskkill', ['/T', '/F', '/PID', String(serverProcess.pid)], {
                    shell: true,
                });
            } else {
                // On Unix, negative PID kills the entire process group
                try {
                    process.kill(-serverProcess.pid, 'SIGTERM');
                } catch {
                    serverProcess.kill('SIGTERM');
                }
            }

            console.log('Backend server stop signal sent');
        } catch (err) {
            console.error(`Error stopping backend: ${err}`);
            // Last resort
            serverProcess.kill();
        }

        serverProcess = null;
    }

    // Verify it actually stopped
    const config = getConfig();
    const serverUrl = `http://localhost:${config.serverPort}`;
    setTimeout(async () => {
        try {
            await axios.get(`${serverUrl}/`, { timeout: 1000 });
            console.warn('Backend server is still running after stop attempt!');
        } catch {
            console.log('Backend server confirmed stopped');
        }
    }, 2000);
}
