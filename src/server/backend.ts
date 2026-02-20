import * as vscode from 'vscode';
import { spawn, ChildProcess } from 'child_process';
import axios from 'axios';
import { getConfig } from '../config/settings';
import { updateStatusBar, resetStatusBar, setStatusBarError } from './statusBar';
import { showError } from './errors';

let serverProcess: ChildProcess | null = null;

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

    const pythonPath =
        vscode.workspace.getConfiguration('python').get<string>('defaultInterpreterPath') || 'python3';
    const backendPath = context.asAbsolutePath('python_backend');

    console.log(`Starting backend at: ${backendPath} on port ${serverPort}`);
    updateStatusBar('$(sync~spin) Starting server...', 'Initializing Python backend');

    serverProcess = spawn(pythonPath, ['-m', 'uvicorn', 'bias_server:app', '--port', String(serverPort)], {
        cwd: backendPath,
        shell: true,
    });

    serverProcess.stdout?.on('data', (data) => {
        console.log(`Backend: ${data}`);
    });

    serverProcess.stderr?.on('data', (data) => {
        console.error(`Backend Error: ${data}`);
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
    if (serverProcess) {
        serverProcess.kill();
        console.log('Backend server stopped');
    }
}
