import * as vscode from 'vscode';
import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import axios from 'axios';
import { getConfig } from '../config/settings';
import { updateStatusBar, resetStatusBar, setStatusBarError } from './statusBar';
import { showError } from './errors';

let serverProcess: ChildProcess | null = null;

/**
 * Resolve the Python executable to use for the backend.
 * Priority: python_backend/venv -> workspace interpreter -> python3
 */
function resolvePythonPath(backendPath: string): string {
    // Check for venv inside python_backend (Windows and Unix paths)
    const venvPythonWin = path.join(backendPath, 'venv', 'Scripts', 'python.exe');
    const venvPythonUnix = path.join(backendPath, 'venv', 'bin', 'python');

    if (fs.existsSync(venvPythonWin)) {
        console.log(`Using venv Python: ${venvPythonWin}`);
        return venvPythonWin;
    }
    if (fs.existsSync(venvPythonUnix)) {
        console.log(`Using venv Python: ${venvPythonUnix}`);
        return venvPythonUnix;
    }

    // Fall back to workspace configured interpreter or system python
    const configured = vscode.workspace.getConfiguration('python').get<string>('defaultInterpreterPath') || 'python3';
    console.log(`No venv found in python_backend, falling back to: ${configured}`);
    return configured;
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
    const pythonPath = resolvePythonPath(backendPath);

    console.log(`Starting backend at: ${backendPath} on port ${serverPort}`);
    console.log(`Using Python: ${pythonPath}`);
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
        if (msg.includes('ERROR') || msg.includes('ModuleNotFoundError') || msg.includes('Traceback')) {
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
            showError(
                'Server Startup Failed',
                'The backend server failed to start. Please check:\n' +
                    '1. Python 3.8+ is installed\n' +
                    '2. Required packages are installed (pip install -r requirements.txt)\n' +
                    `3. Port ${serverPort} is not in use`,
            );
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
