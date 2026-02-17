// src/extension.ts
import * as vscode from 'vscode';
import { spawn, ChildProcess } from 'child_process';
import axios, { AxiosError } from 'axios';
import * as path from 'path';
import * as fs from 'fs';

let serverProcess: ChildProcess | null = null;
let statusBarItem: vscode.StatusBarItem;

// Configuration helper
function getConfig() {
    const config = vscode.workspace.getConfiguration('fairlint-dl');
    return {
        // Training settings
        epochs: config.get<number>('training.epochs', 30),
        batchSize: config.get<number>('training.batchSize', 32),
        hiddenLayers: config.get<string>('training.hiddenLayers', '64,32,16,8,4'),
        // Analysis settings
        qidThreshold: config.get<number>('analysis.qidThreshold', 0.1),
        maxSamples: config.get<number>('analysis.maxSamples', 500),
        // Search settings
        globalIterations: config.get<number>('search.globalIterations', 50),
        localNeighbors: config.get<number>('search.localNeighbors', 30),
        // Server settings
        serverPort: config.get<number>('server.port', 8765),
        // Detection settings
        autoDetectProtected: config.get<boolean>('detection.autoDetectProtected', true),
        // Visualization settings
        showCharts: config.get<boolean>('visualization.showCharts', true),
        theme: config.get<string>('visualization.theme', 'dark'),
        // Notification settings
        showProgress: config.get<boolean>('notifications.showProgress', true),
        showStatusBar: config.get<boolean>('notifications.showStatusBar', true),
    };
}

// Get server URL from config
function getServerUrl(): string {
    const config = getConfig();
    return `http://localhost:${config.serverPort}`;
}

// Default server port (can be changed via settings)
const DEFAULT_SERVER_PORT = 8765;

// Error message mappings for user-friendly messages
const ERROR_MESSAGES: Record<string, string> = {
    ECONNREFUSED: 'Cannot connect to the analysis server. Please wait for it to start or restart VS Code.',
    ETIMEDOUT: 'The analysis is taking longer than expected. This may happen with large datasets.',
    ENOTFOUND: 'Network error. Please check your connection.',
    'Train model first': 'Please run training first before performing analysis.',
};

export async function activate(context: vscode.ExtensionContext) {
    console.log('Fairness DL Extension activating...');

    // Initialize status bar item
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    statusBarItem.text = '$(pulse) FairLint-DL';
    statusBarItem.tooltip = 'Click to analyze a dataset for fairness';
    statusBarItem.command = 'fairlint-dl.analyzeDataset';
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    // Start Python backend
    await startBackend(context);

    // Register commands
    context.subscriptions.push(vscode.commands.registerCommand('fairlint-dl.analyzeDataset', analyzeDataset));

    // Register file context menu
    context.subscriptions.push(
        vscode.commands.registerCommand('fairlint-dl.analyzeFromMenu', async (uri: vscode.Uri) => {
            await analyzeDataset(uri);
        }),
    );
}

async function startBackend(context: vscode.ExtensionContext) {
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
        // Server not running, proceed to spawn
        console.log(`Server not running, starting new instance...`);
    }

    const pythonPath = vscode.workspace.getConfiguration('python').get<string>('defaultInterpreterPath') || 'python3';
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

    // Show startup progress
    await vscode.window.withProgress(
        {
            location: vscode.ProgressLocation.Notification,
            title: 'Starting Fairness Analysis Server...',
            cancellable: false,
        },
        async (progress) => {
            progress.report({ message: 'Initializing Python backend...' });

            // Wait for server to start with retries
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

/**
 * Show a user-friendly error message with optional details
 */
function showError(title: string, detail: string, actions?: { label: string; action: () => void }[]) {
    const message = `${title}: ${detail}`;

    if (actions && actions.length > 0) {
        const actionLabels = actions.map((a) => a.label);
        vscode.window.showErrorMessage(message, ...actionLabels).then((selected) => {
            const action = actions.find((a) => a.label === selected);
            if (action) {
                action.action();
            }
        });
    } else {
        vscode.window.showErrorMessage(message);
    }
}

/**
 * Parse error and return user-friendly message
 */
function parseError(error: any): { title: string; detail: string; suggestion: string } {
    if (axios.isAxiosError(error)) {
        const axiosError = error as AxiosError;

        // Network errors
        if (axiosError.code) {
            const friendlyMessage = ERROR_MESSAGES[axiosError.code];
            if (friendlyMessage) {
                return {
                    title: 'Connection Error',
                    detail: friendlyMessage,
                    suggestion: 'Try restarting VS Code or checking if the server is running.',
                };
            }
        }

        // HTTP errors
        if (axiosError.response) {
            const status = axiosError.response.status;
            const data = axiosError.response.data as any;
            const detail = data?.detail || axiosError.message;

            switch (status) {
                case 400:
                    return {
                        title: 'Invalid Request',
                        detail: detail,
                        suggestion: 'Please check your input and try again.',
                    };
                case 404:
                    return {
                        title: 'Not Found',
                        detail: detail,
                        suggestion: 'The requested file or resource was not found.',
                    };
                case 500:
                    // Parse common backend errors
                    if (detail.includes('label_column')) {
                        return {
                            title: 'Invalid Column',
                            detail: 'The specified label column was not found in the dataset.',
                            suggestion: 'Please select a valid column from the dropdown.',
                        };
                    }
                    if (detail.includes('memory')) {
                        return {
                            title: 'Memory Error',
                            detail: 'The dataset is too large to process.',
                            suggestion: 'Try using a smaller dataset or increasing available memory.',
                        };
                    }
                    return {
                        title: 'Analysis Error',
                        detail: detail,
                        suggestion: 'Check the console for more details.',
                    };
                default:
                    return {
                        title: `Server Error (${status})`,
                        detail: detail,
                        suggestion: 'An unexpected error occurred.',
                    };
            }
        }

        // Timeout
        if (axiosError.code === 'ECONNABORTED') {
            return {
                title: 'Timeout',
                detail: 'The operation took too long to complete.',
                suggestion: 'Try with a smaller dataset or fewer epochs.',
            };
        }
    }

    // Generic error
    return {
        title: 'Error',
        detail: error.message || 'An unknown error occurred.',
        suggestion: 'Please try again or check the console for details.',
    };
}

/**
 * Status bar update helpers
 */
function updateStatusBar(text: string, tooltip?: string, color?: string) {
    const config = getConfig();
    if (!config.showStatusBar) {
        return;
    }

    statusBarItem.text = text;
    if (tooltip) {
        statusBarItem.tooltip = tooltip;
    }
    if (color) {
        statusBarItem.backgroundColor = new vscode.ThemeColor(color);
    } else {
        statusBarItem.backgroundColor = undefined;
    }
    statusBarItem.show();
}

function resetStatusBar() {
    statusBarItem.text = '$(pulse) FairLint-DL';
    statusBarItem.tooltip = 'Click to analyze a dataset for fairness';
    statusBarItem.backgroundColor = undefined;
    statusBarItem.show();
}

function setStatusBarSuccess(message: string) {
    updateStatusBar(`$(check) ${message}`, 'Analysis complete - click to analyze another dataset');
    setTimeout(resetStatusBar, 5000);
}

function setStatusBarError(message: string) {
    updateStatusBar(`$(error) ${message}`, 'Analysis failed - click to try again', 'statusBarItem.errorBackground');
    setTimeout(resetStatusBar, 8000);
}

/**
 * Fetch column names from CSV file
 */
async function fetchColumns(
    filePath: string,
): Promise<{ columns: string[]; sampleData: any[]; detectedSensitive: string[] } | null> {
    const serverUrl = getServerUrl();
    try {
        const response = await axios.post(`${serverUrl}/columns`, { file_path: filePath }, { timeout: 10000 });

        return {
            columns: response.data.columns,
            sampleData: response.data.sample_data,
            detectedSensitive: response.data.detected_sensitive || [],
        };
    } catch (error) {
        const parsed = parseError(error);
        showError(parsed.title, `${parsed.detail}\n${parsed.suggestion}`);
        return null;
    }
}

async function analyzeDataset(uri: vscode.Uri) {
    const filePath = uri.fsPath;

    // Verify it's a CSV
    if (!filePath.endsWith('.csv')) {
        showError('Invalid File Type', 'Please select a CSV file (.csv extension required).');
        return;
    }

    // Verify file exists
    if (!fs.existsSync(filePath)) {
        showError('File Not Found', `The file "${path.basename(filePath)}" does not exist.`);
        return;
    }

    // Fetch column names with progress
    let columnData: { columns: string[]; sampleData: any[] } | null = null;

    await vscode.window.withProgress(
        {
            location: vscode.ProgressLocation.Notification,
            title: 'Loading CSV columns...',
            cancellable: false,
        },
        async () => {
            columnData = await fetchColumns(filePath);
        },
    );

    if (
        !columnData ||
        (columnData as { columns: string[]; sampleData: any[]; detectedSensitive: string[] }).columns.length === 0
    ) {
        showError('Empty Dataset', 'The CSV file appears to be empty or has no columns.');
        return;
    }

    // Cast to proper type after null check
    const validColumnData = columnData as { columns: string[]; sampleData: any[]; detectedSensitive: string[] };

    // Create QuickPick items with column info
    const quickPickItems: vscode.QuickPickItem[] = validColumnData.columns.map((col: string, index: number) => {
        // Get sample value from first row if available
        const sampleValue = validColumnData.sampleData[0]?.[col];
        const sampleStr = sampleValue !== undefined ? ` (e.g., "${sampleValue}")` : '';

        return {
            label: col,
            description: `Column ${index + 1}${sampleStr}`,
            detail: `Select this column as the target/label variable`,
        };
    });

    // Show QuickPick for column selection
    const selectedColumn = await vscode.window.showQuickPick(quickPickItems, {
        placeHolder: 'Select the target/label column for prediction',
        title: 'Fairness Analysis: Select Label Column',
        matchOnDescription: true,
        matchOnDetail: false,
    });

    if (!selectedColumn) {
        return; // User cancelled
    }

    const labelColumn = selectedColumn.label;

    // Step 2: Select protected attributes (multi-select with auto-detected pre-selected)
    const protectedItems: vscode.QuickPickItem[] = validColumnData.columns
        .filter((col: string) => col !== labelColumn)
        .map((col: string) => ({
            label: col,
            picked: validColumnData.detectedSensitive.includes(col),
            description: validColumnData.detectedSensitive.includes(col) ? '(auto-detected)' : '',
        }));

    const selectedProtected = await vscode.window.showQuickPick(protectedItems, {
        canPickMany: true,
        placeHolder: 'Select protected/sensitive attributes (auto-detected ones are pre-selected)',
        title: 'Fairness Analysis: Protected Attributes',
    });

    if (!selectedProtected || selectedProtected.length === 0) {
        showError('No Protected Attributes', 'At least one protected attribute must be selected.');
        return;
    }

    const protectedFeatures = selectedProtected.map((item) => item.label);

    // Step 3: Select DNN architecture
    const defaultLayers = getConfig().hiddenLayers;
    const archChoice = await vscode.window.showQuickPick(
        [
            {
                label: `Default (${defaultLayers})`,
                description: 'DICE paper architecture',
                detail: `Hidden layers: [${defaultLayers}] → Output(2)`,
                value: defaultLayers,
            },
            {
                label: 'Wide (128,64,32,16)',
                description: '4-layer wider network',
                detail: 'Hidden layers: [128, 64, 32, 16] → Output(2)',
                value: '128,64,32,16',
            },
            {
                label: 'Deep (256,128,64,32,16,8)',
                description: '6-layer deep network',
                detail: 'Hidden layers: [256, 128, 64, 32, 16, 8] → Output(2)',
                value: '256,128,64,32,16,8',
            },
            {
                label: 'Custom...',
                description: 'Enter custom layer sizes',
                detail: 'Specify comma-separated hidden layer sizes',
                value: 'custom',
            },
        ],
        {
            placeHolder: 'Select DNN architecture for fairness analysis',
            title: 'Fairness Analysis: Model Architecture',
        },
    );

    if (!archChoice) {
        return; // User cancelled
    }

    let hiddenLayersStr = (archChoice as any).value as string;

    if (hiddenLayersStr === 'custom') {
        const customInput = await vscode.window.showInputBox({
            prompt: 'Enter hidden layer sizes (comma-separated, e.g., 128,64,32)',
            value: defaultLayers,
            validateInput: (value) => {
                const nums = value.split(',').map((s) => parseInt(s.trim(), 10));
                if (nums.some((n) => isNaN(n) || n <= 0)) {
                    return 'All values must be positive integers';
                }
                if (nums.length < 2) {
                    return 'At least 2 hidden layers required';
                }
                return null;
            },
        });

        if (!customInput) {
            return; // User cancelled
        }
        hiddenLayersStr = customInput;
    }

    const hiddenLayers = hiddenLayersStr
        .split(',')
        .map((s: string) => parseInt(s.trim(), 10))
        .filter((n: number) => !isNaN(n) && n > 0);

    // Run the analysis with improved progress
    await runAnalysis(filePath, labelColumn, protectedFeatures, hiddenLayers);
}

async function runAnalysis(filePath: string, labelColumn: string, protectedFeatures: string[], hiddenLayers: number[]) {
    const config = getConfig();
    const serverUrl = getServerUrl();

    await vscode.window.withProgress(
        {
            location: config.showProgress ? vscode.ProgressLocation.Notification : vscode.ProgressLocation.Window,
            title: 'Fairness Analysis',
            cancellable: false,
        },
        async (progress) => {
            const startTime = Date.now();
            const fileName = path.basename(filePath);

            try {
                // Step 1/6: Train model (0-30%)
                progress.report({
                    increment: 0,
                    message: 'Step 1/6: Training neural network model...',
                });
                updateStatusBar('$(sync~spin) Training DNN...', `Training on ${fileName}`);

                const trainResponse = await axios.post(
                    `${serverUrl}/train`,
                    {
                        file_path: filePath,
                        label_column: labelColumn,
                        sensitive_features: protectedFeatures,
                        num_epochs: config.epochs,
                        batch_size: config.batchSize,
                        hidden_layers: hiddenLayers,
                    },
                    { timeout: 300000 },
                );

                const trainTime = Math.round((Date.now() - startTime) / 1000);
                progress.report({
                    increment: 30,
                    message: `Step 1/6: Training complete (${trainTime}s) - ${trainResponse.data.accuracy.toFixed(
                        1,
                    )}% accuracy`,
                });
                updateStatusBar(
                    `$(check) Trained [${trainResponse.data.accuracy.toFixed(0)}%]`,
                    `Training complete: ${trainResponse.data.accuracy.toFixed(1)}% accuracy`,
                );

                // Step 2/6: Internal space visualization (30-38%)
                progress.report({
                    increment: 0,
                    message: 'Step 2/6: Computing internal space visualization...',
                });
                updateStatusBar('$(eye) Computing activations...', 'Dimensionality reduction');

                const activationsResponse = await axios.post(
                    `${serverUrl}/activations`,
                    { method: 'pca', max_samples: config.maxSamples },
                    { timeout: 60000 },
                );

                progress.report({
                    increment: 8,
                    message: 'Step 2/6: Internal space computed',
                });

                // Step 3/6: QID Analysis (38-55%)
                progress.report({
                    increment: 0,
                    message: 'Step 3/6: Computing fairness metrics (QID analysis)...',
                });
                updateStatusBar('$(beaker) Computing QID...', 'Calculating fairness metrics');

                const protectedValues: Record<string, number[]> = {};
                protectedFeatures.forEach((_feature: string, idx: number) => {
                    protectedValues[idx.toString()] = [0.0, 1.0];
                });

                const analyzeResponse = await axios.post(
                    `${serverUrl}/analyze`,
                    {
                        file_path: filePath,
                        label_column: labelColumn,
                        sensitive_features: protectedFeatures,
                        protected_values: protectedValues,
                        max_samples: config.maxSamples,
                        qid_threshold: config.qidThreshold,
                    },
                    { timeout: 120000 },
                );

                progress.report({
                    increment: 17,
                    message: `Step 3/6: QID metrics computed - Mean QID: ${analyzeResponse.data.qid_metrics.mean_qid.toFixed(
                        4,
                    )} bits`,
                });
                updateStatusBar(
                    `$(graph) QID: ${analyzeResponse.data.qid_metrics.mean_qid.toFixed(2)} bits`,
                    `Mean QID: ${analyzeResponse.data.qid_metrics.mean_qid.toFixed(4)} bits`,
                );

                // Step 4/6: Search (55-70%)
                progress.report({
                    increment: 0,
                    message: 'Step 4/6: Searching for discriminatory instances...',
                });
                updateStatusBar('$(search) Searching...', 'Finding discriminatory instances');

                const searchResponse = await axios.post(
                    `${serverUrl}/search`,
                    {
                        protected_values: protectedValues,
                        num_iterations: config.globalIterations,
                        num_neighbors: config.localNeighbors,
                    },
                    { timeout: 120000 },
                );

                progress.report({
                    increment: 15,
                    message: `Step 4/6: Found ${searchResponse.data.search_results.num_found} discriminatory instances`,
                });
                updateStatusBar(
                    `$(bug) Found ${searchResponse.data.search_results.num_found}`,
                    `Found ${searchResponse.data.search_results.num_found} discriminatory instances`,
                );

                // Step 5/6: Debug (70-82%)
                progress.report({
                    increment: 0,
                    message: 'Step 5/6: Localizing biased layers and neurons...',
                });
                updateStatusBar('$(telescope) Debugging...', 'Localizing biased neurons');

                const debugResponse = await axios.post(
                    `${serverUrl}/debug`,
                    {
                        protected_values: protectedValues,
                        num_iterations: config.globalIterations,
                        num_neighbors: config.localNeighbors,
                    },
                    { timeout: 120000 },
                );

                progress.report({
                    increment: 12,
                    message: 'Step 5/6: Causal debugging complete',
                });

                // Step 6/6: LIME & SHAP Explanations (82-100%)
                progress.report({
                    increment: 0,
                    message: 'Step 6/6: Computing LIME & SHAP explanations...',
                });
                updateStatusBar('$(lightbulb) Computing explanations...', 'Running LIME and SHAP');

                const explainResponse = await axios.post(
                    `${serverUrl}/explain`,
                    { method: 'both', num_instances: 10, max_background: 100 },
                    { timeout: 180000 },
                );

                const totalTime = Math.round((Date.now() - startTime) / 1000);
                progress.report({
                    increment: 18,
                    message: `Analysis complete! Total time: ${totalTime}s`,
                });

                // Brief pause to show completion
                await new Promise((resolve) => setTimeout(resolve, 500));

                // Show results
                showResults({
                    training: trainResponse.data,
                    analysis: analyzeResponse.data,
                    search: searchResponse.data,
                    debug: debugResponse.data,
                    activations: activationsResponse.data.activations,
                    explanations: explainResponse.data.explanations,
                    metadata: {
                        file: fileName,
                        filePath: filePath,
                        labelColumn: labelColumn,
                        totalTime: totalTime,
                    },
                    config: config,
                });

                // Update status bar with success
                setStatusBarSuccess(`Done [${analyzeResponse.data.qid_metrics.num_discriminatory} issues]`);

                vscode.window.showInformationMessage(
                    `Fairness analysis complete! Found ${analyzeResponse.data.qid_metrics.num_discriminatory} potentially discriminatory instances.`,
                );
            } catch (error: any) {
                console.error('Analysis error:', error);
                const parsed = parseError(error);
                showError(parsed.title, `${parsed.detail}\n\nSuggestion: ${parsed.suggestion}`);
                setStatusBarError('Analysis failed');
            }
        },
    );
}

function showResults(results: any) {
    const panel = vscode.window.createWebviewPanel(
        'fairnessResults',
        'Fairness Analysis Results',
        vscode.ViewColumn.One,
        {
            enableScripts: true,
            retainContextWhenHidden: true,
        },
    );

    const html = getWebviewHtml(results);
    panel.webview.html = html;

    // Handle messages from the webview (for export)
    panel.webview.onDidReceiveMessage(async (message) => {
        if (message.command === 'saveJson') {
            const baseFileName = results.metadata?.file?.replace('.csv', '') || 'fairness-report';
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
            const defaultDir = results.metadata?.filePath
                ? path.dirname(results.metadata.filePath)
                : vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '';
            const defaultName = `${baseFileName}_data_${timestamp}.json`;
            const defaultUri = defaultDir
                ? vscode.Uri.file(path.join(defaultDir, defaultName))
                : undefined;

            const uri = await vscode.window.showSaveDialog({
                defaultUri,
                filters: { 'JSON Files': ['json'], 'All Files': ['*'] },
                title: 'Save Analysis Data (JSON)',
            });

            if (uri) {
                try {
                    const exportData = {
                        generated_at: new Date().toISOString(),
                        metadata: results.metadata,
                        training: {
                            accuracy: results.training.accuracy,
                            num_parameters: results.training.num_parameters,
                            protected_features: results.training.protected_features,
                        },
                        qid_metrics: results.analysis.qid_metrics,
                        search_results: {
                            best_qid: results.search.search_results.best_qid,
                            num_found: results.search.search_results.num_found,
                        },
                        layer_analysis: results.debug?.layer_analysis,
                        neuron_analysis: results.debug?.neuron_analysis,
                        explanations: {
                            shap_global_importance: results.explanations?.shap?.global_importance,
                            shap_feature_names: results.explanations?.shap?.feature_names,
                            lime_aggregated_importance: results.explanations?.lime?.aggregated_importance,
                            lime_feature_names: results.explanations?.lime?.feature_names,
                        },
                    };
                    fs.writeFileSync(uri.fsPath, JSON.stringify(exportData, null, 2), 'utf-8');
                    vscode.window.showInformationMessage(`Data exported to ${path.basename(uri.fsPath)}`);
                } catch (err: any) {
                    vscode.window.showErrorMessage(`Failed to export data: ${err.message}`);
                }
            }
        }
    });
}

function getWebviewHtml(results: any): string {
    const qidMetrics = results.analysis.qid_metrics;
    const searchResults = results.search.search_results;
    const layerAnalysis = results.debug?.layer_analysis;
    const neuronAnalysis = results.debug?.neuron_analysis;
    const activationsData = results.activations || null;
    const explanationsData = results.explanations || null;
    const metadata = results.metadata;

    // Determine overall fairness status
    const fairnessScore = calculateFairnessScore(qidMetrics);
    const fairnessStatus = getFairnessStatus(fairnessScore);

    return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script>
        // VS Code API - must be acquired before any other scripts
        var _vscodeApi = (function() { try { return acquireVsCodeApi(); } catch(e) { return null; } })();

        function downloadJson() {
            if (!_vscodeApi) return;
            try {
                _vscodeApi.postMessage({ command: 'saveJson' });
            } catch (e) {
                console.error('Download JSON error:', e);
            }
        }
    </script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        :root {
            --bg-primary: #1e1e1e;
            --bg-secondary: #252526;
            --bg-tertiary: #2d2d30;
            --bg-hover: #3c3c3c;
            --text-primary: #e4e4e4;
            --text-secondary: #a0a0a0;
            --text-muted: #6e6e6e;
            --accent-blue: #4fc3f7;
            --accent-green: #4caf50;
            --accent-yellow: #ffc107;
            --accent-orange: #ff9800;
            --accent-red: #f44336;
            --accent-purple: #9c27b0;
            --border-color: #3c3c3c;
            --shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            --transition: all 0.2s ease;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 24px;
            max-width: 1400px;
            margin: 0 auto;
        }

        /* Header */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 32px;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--border-color);
        }

        .header-left h1 {
            font-size: 28px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 8px;
        }

        .header-meta {
            display: flex;
            gap: 24px;
            color: var(--text-secondary);
            font-size: 14px;
        }

        .header-meta span {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        /* Overall Score Card */
        .score-card {
            background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            min-width: 200px;
            border: 1px solid var(--border-color);
        }

        .score-value {
            font-size: 48px;
            font-weight: 700;
            margin-bottom: 4px;
        }

        .score-label {
            font-size: 14px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .score-status {
            margin-top: 12px;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 500;
        }

        .status-good { background: rgba(76, 175, 80, 0.2); color: var(--accent-green); }
        .status-warning { background: rgba(255, 152, 0, 0.2); color: var(--accent-orange); }
        .status-danger { background: rgba(244, 67, 54, 0.2); color: var(--accent-red); }

        /* Section */
        .section {
            margin-bottom: 32px;
        }

        .section-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 16px;
        }

        .section-icon {
            width: 32px;
            height: 32px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
        }

        .section-title {
            font-size: 20px;
            font-weight: 600;
            color: var(--text-primary);
        }

        /* Cards Grid */
        .cards-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
        }

        .card {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            transition: var(--transition);
        }

        .card:hover {
            background: var(--bg-tertiary);
            transform: translateY(-2px);
            box-shadow: var(--shadow);
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 12px;
        }

        .card-label {
            font-size: 13px;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .card-badge {
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }

        .badge-success { background: rgba(76, 175, 80, 0.2); color: var(--accent-green); }
        .badge-warning { background: rgba(255, 152, 0, 0.2); color: var(--accent-orange); }
        .badge-danger { background: rgba(244, 67, 54, 0.2); color: var(--accent-red); }
        .badge-info { background: rgba(79, 195, 247, 0.2); color: var(--accent-blue); }

        .card-value {
            font-size: 32px;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 4px;
        }

        .card-unit {
            font-size: 16px;
            font-weight: 400;
            color: var(--text-secondary);
            margin-left: 4px;
        }

        .card-description {
            font-size: 13px;
            color: var(--text-muted);
            margin-top: 8px;
        }

        /* Progress Bar */
        .progress-container {
            margin-top: 12px;
        }

        .progress-bar {
            height: 6px;
            background: var(--bg-primary);
            border-radius: 3px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.3s ease;
        }

        .progress-labels {
            display: flex;
            justify-content: space-between;
            margin-top: 4px;
            font-size: 11px;
            color: var(--text-muted);
        }

        /* Chart Container */
        .chart-container {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            margin-top: 16px;
        }

        .chart-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--text-primary);
        }

        /* Neuron List */
        .neuron-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .neuron-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 16px;
            background: var(--bg-tertiary);
            border-radius: 8px;
            transition: var(--transition);
        }

        .neuron-item:hover {
            background: var(--bg-hover);
        }

        .neuron-info {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .neuron-rank {
            width: 28px;
            height: 28px;
            background: var(--accent-blue);
            color: var(--bg-primary);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: 700;
        }

        .neuron-name {
            font-weight: 500;
        }

        .neuron-score {
            font-weight: 600;
            color: var(--accent-orange);
        }

        /* Interpretation Box */
        .interpretation-box {
            background: linear-gradient(135deg, rgba(79, 195, 247, 0.1) 0%, rgba(156, 39, 176, 0.1) 100%);
            border: 1px solid rgba(79, 195, 247, 0.3);
            border-radius: 12px;
            padding: 20px;
            margin-top: 16px;
        }

        .interpretation-title {
            font-size: 14px;
            font-weight: 600;
            color: var(--accent-blue);
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .interpretation-content {
            font-size: 14px;
            color: var(--text-secondary);
            line-height: 1.7;
        }

        .interpretation-content strong {
            color: var(--text-primary);
        }

        /* Legal Compliance Box */
        .compliance-box {
            padding: 16px 20px;
            border-radius: 12px;
            margin-top: 16px;
        }

        .compliance-pass {
            background: rgba(76, 175, 80, 0.1);
            border: 1px solid rgba(76, 175, 80, 0.3);
        }

        .compliance-fail {
            background: rgba(244, 67, 54, 0.1);
            border: 1px solid rgba(244, 67, 54, 0.3);
        }

        .compliance-header {
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 600;
            margin-bottom: 8px;
        }

        .compliance-pass .compliance-header { color: var(--accent-green); }
        .compliance-fail .compliance-header { color: var(--accent-red); }

        .compliance-text {
            font-size: 13px;
            color: var(--text-secondary);
        }

        /* Export Buttons */
        .export-toolbar {
            display: flex;
            gap: 8px;
            align-items: center;
            margin-top: 12px;
        }

        .export-btn {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 8px 14px;
            border: 1px solid var(--border);
            border-radius: 8px;
            background: var(--card-bg);
            color: var(--text-secondary);
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            font-family: inherit;
        }

        .export-btn:hover {
            background: var(--accent-blue);
            color: #fff;
            border-color: var(--accent-blue);
            transform: translateY(-1px);
        }

        .export-btn svg {
            flex-shrink: 0;
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .section {
            animation: fadeIn 0.4s ease forwards;
        }

        .section:nth-child(2) { animation-delay: 0.1s; }
        .section:nth-child(3) { animation-delay: 0.2s; }
        .section:nth-child(4) { animation-delay: 0.3s; }
        .section:nth-child(5) { animation-delay: 0.4s; }

    </style>
</head>
<body>
    <!-- Header -->
    <div class="header">
        <div class="header-left">
            <h1>Fairness Analysis Results</h1>
            <div class="header-meta">
                <span>
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M4 0h8v1H4V0zM2 1h2v1H2V1zm10 0h2v1h-2V1zM1 2h1v1H1V2zm12 0h2v1h-2V2zM1 14h1v1H1v-1zm12 0h2v1h-2v-1zM2 14h2v1H2v-1zm10 0h2v1h-2v-1zM4 15h8v1H4v-1zM0 3h1v11H0V3zm15 0h1v11h-1V3z"/></svg>
                    ${metadata?.file || 'Dataset'}
                </span>
                <span>
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0zm0 14A6 6 0 1 1 8 2a6 6 0 0 1 0 12zm-.5-9h1v4h-1V5zm0 5h1v1h-1v-1z"/></svg>
                    Label: ${metadata?.labelColumn || 'N/A'}
                </span>
                <span>
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0zm0 14A6 6 0 1 1 8 2a6 6 0 0 1 0 12zM7 4h2v5H7V4zm0 6h2v2H7v-2z"/></svg>
                    ${metadata?.totalTime || 0}s analysis time
                </span>
            </div>
            <div class="export-toolbar">
                <button class="export-btn" onclick="downloadJson()" title="Download raw analysis data as JSON">
                    <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M5 0C3.9 0 3 .9 3 2v12c0 1.1.9 2 2 2h6c1.1 0 2-.9 2-2V6l-5-6H5zm0 1h3v4h4v9H5V1zm4 0l3 3H9V1z"/></svg>
                    Export JSON
                </button>
            </div>
        </div>
        <div class="score-card">
            <div class="score-value" style="color: ${fairnessStatus.color}">${fairnessScore}</div>
            <div class="score-label">Fairness Score</div>
            <div class="score-status ${fairnessStatus.class}">${fairnessStatus.label}</div>
        </div>
    </div>

    <!-- Model Training Section -->
    <div class="section">
        <div class="section-header">
            <div class="section-icon" style="background: rgba(76, 175, 80, 0.2);">
                <span style="color: var(--accent-green);">📊</span>
            </div>
            <h2 class="section-title">Model Training</h2>
        </div>
        <div class="cards-grid">
            <div class="card">
                <div class="card-header">
                    <span class="card-label">Model Accuracy</span>
                    <span class="card-badge badge-success">Trained</span>
                </div>
                <div class="card-value">${results.training.accuracy.toFixed(1)}<span class="card-unit">%</span></div>
                <div class="card-description">Test set classification accuracy</div>
                <div class="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${
                            results.training.accuracy
                        }%; background: var(--accent-green);"></div>
                    </div>
                </div>
            </div>
            <div class="card">
                <div class="card-header">
                    <span class="card-label">Protected Features</span>
                    <span class="card-badge badge-info">User-Selected</span>
                </div>
                <div class="card-value">${results.training.protected_features.length}</div>
                <div class="card-description">${results.training.protected_features.join(', ')}</div>
            </div>
            <div class="card">
                <div class="card-header">
                    <span class="card-label">Model Size</span>
                </div>
                <div class="card-value">${(results.training.num_parameters / 1000).toFixed(
                    1,
                )}<span class="card-unit">K params</span></div>
                <div class="card-description">Total trainable parameters</div>
            </div>
        </div>
    </div>

    ${
        activationsData
            ? `
    <!-- Internal Space Visualization Section -->
    <div class="section">
        <div class="section-header">
            <div class="section-icon" style="background: rgba(156, 39, 176, 0.2);">
                <span style="color: var(--accent-purple);">🔬</span>
            </div>
            <h2 class="section-title">Internal Space Visualization</h2>
        </div>
        <div class="interpretation-box" style="margin-bottom: 16px;">
            <div class="interpretation-title">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1zm0 12.5a5.5 5.5 0 1 1 0-11 5.5 5.5 0 0 1 0 11zM7 4h2v5H7V4zm0 6h2v2H7v-2z"/></svg>
                How to Read These Plots
            </div>
            <div class="interpretation-content">
                Each chart shows the <strong>${activationsData.method.toUpperCase()}</strong> 2D projection of layer activations for <strong>${activationsData.num_samples}</strong> test instances.
                Points are colored by <strong>prediction label</strong>. Clusters that separate by protected attribute indicate the model is learning to encode protected information at that layer.
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 16px;">
            ${activationsData.layers
                .map(
                    (layer: any, idx: number) => `
                <div class="chart-container">
                    <div class="chart-title">${layer.layer_name} Activations (${activationsData.method.toUpperCase()})</div>
                    <div id="activation-layer-${idx}" style="height: 350px;"></div>
                </div>
            `,
                )
                .join('')}
        </div>
    </div>
    `
            : ''
    }

    <!-- QID Metrics Section -->
    <div class="section">
        <div class="section-header">
            <div class="section-icon" style="background: rgba(79, 195, 247, 0.2);">
                <span style="color: var(--accent-blue);">⚖️</span>
            </div>
            <h2 class="section-title">Fairness Metrics (QID Analysis)</h2>
        </div>
        <div class="cards-grid">
            <div class="card">
                <div class="card-header">
                    <span class="card-label">Mean QID</span>
                    <span class="card-badge ${qidMetrics.mean_qid > 1.0 ? 'badge-warning' : 'badge-success'}">${
                        qidMetrics.mean_qid > 1.0 ? 'High' : 'Low'
                    }</span>
                </div>
                <div class="card-value">${qidMetrics.mean_qid.toFixed(4)}<span class="card-unit">bits</span></div>
                <div class="card-description">Average protected information used in decisions. Lower is better.</div>
            </div>
            <div class="card">
                <div class="card-header">
                    <span class="card-label">Max QID</span>
                    <span class="card-badge ${qidMetrics.max_qid > 2.0 ? 'badge-danger' : 'badge-success'}">${
                        qidMetrics.max_qid > 2.0 ? 'Critical' : 'Normal'
                    }</span>
                </div>
                <div class="card-value">${qidMetrics.max_qid.toFixed(4)}<span class="card-unit">bits</span></div>
                <div class="card-description">Worst-case protected information leakage.</div>
            </div>
            <div class="card">
                <div class="card-header">
                    <span class="card-label">Discriminatory Instances</span>
                    <span class="card-badge ${
                        qidMetrics.pct_discriminatory > 10 ? 'badge-warning' : 'badge-success'
                    }">${qidMetrics.pct_discriminatory > 10 ? 'Concerning' : 'Acceptable'}</span>
                </div>
                <div class="card-value">${
                    qidMetrics.num_discriminatory
                }<span class="card-unit">(${qidMetrics.pct_discriminatory.toFixed(1)}%)</span></div>
                <div class="card-description">Instances with QID > 0.1 bits showing potential bias.</div>
                <div class="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${Math.min(
                            qidMetrics.pct_discriminatory,
                            100,
                        )}%; background: ${
                            qidMetrics.pct_discriminatory > 10 ? 'var(--accent-orange)' : 'var(--accent-green)'
                        };"></div>
                    </div>
                    <div class="progress-labels">
                        <span>0%</span>
                        <span>10% threshold</span>
                        <span>100%</span>
                    </div>
                </div>
            </div>
            <div class="card">
                <div class="card-header">
                    <span class="card-label">Disparate Impact Ratio</span>
                    <span class="card-badge ${
                        qidMetrics.mean_disparate_impact < 0.8 ? 'badge-danger' : 'badge-success'
                    }">${qidMetrics.mean_disparate_impact < 0.8 ? 'Violation' : 'Compliant'}</span>
                </div>
                <div class="card-value">${qidMetrics.mean_disparate_impact.toFixed(3)}</div>
                <div class="card-description">Ratio should be ≥ 0.8 for legal compliance.</div>
                <div class="compliance-box ${
                    qidMetrics.mean_disparate_impact >= 0.8 ? 'compliance-pass' : 'compliance-fail'
                }">
                    <div class="compliance-header">
                        ${qidMetrics.mean_disparate_impact >= 0.8 ? '✓ Passes 80% Rule' : '✗ Violates 80% Rule'}
                    </div>
                    <div class="compliance-text">
                        ${
                            qidMetrics.mean_disparate_impact >= 0.8
                                ? 'Model meets legal fairness thresholds for hiring/lending decisions.'
                                : 'Model may exhibit legally actionable discrimination. Consider retraining with fairness constraints.'
                        }
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Search Results Section -->
    <div class="section">
        <div class="section-header">
            <div class="section-icon" style="background: rgba(255, 152, 0, 0.2);">
                <span style="color: var(--accent-orange);">🔎</span>
            </div>
            <h2 class="section-title">Discriminatory Instance Search</h2>
        </div>
        <div class="cards-grid">
            <div class="card">
                <div class="card-header">
                    <span class="card-label">Best QID Found</span>
                </div>
                <div class="card-value">${searchResults.best_qid.toFixed(4)}<span class="card-unit">bits</span></div>
                <div class="card-description">Maximum discrimination discovered via gradient search.</div>
            </div>
            <div class="card">
                <div class="card-header">
                    <span class="card-label">Instances Generated</span>
                </div>
                <div class="card-value">${searchResults.num_found}</div>
                <div class="card-description">Discriminatory test cases found in local search.</div>
            </div>
        </div>

        <!-- Charts Row -->
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px;">
            <div class="chart-container">
                <div class="chart-title">QID Distribution Histogram</div>
                <div id="qid-histogram" style="height: 320px;"></div>
            </div>
            <div class="chart-container">
                <div class="chart-title">Disparate Impact Gauge</div>
                <div id="disparate-impact-gauge" style="height: 320px;"></div>
            </div>
        </div>

        <div class="chart-container">
            <div class="chart-title">QID Values by Instance (Scatter Plot)</div>
            <div id="qid-chart" style="height: 350px;"></div>
        </div>
    </div>

    ${
        layerAnalysis
            ? `
    <!-- Layer Analysis Section -->
    <div class="section">
        <div class="section-header">
            <div class="section-icon" style="background: rgba(156, 39, 176, 0.2);">
                <span style="color: var(--accent-purple);">🧠</span>
            </div>
            <h2 class="section-title">Causal Debugging: Layer Analysis</h2>
        </div>
        <div class="cards-grid">
            <div class="card">
                <div class="card-header">
                    <span class="card-label">Most Biased Layer</span>
                    <span class="card-badge ${
                        layerAnalysis.biased_layer.sensitivity > 0.5 ? 'badge-warning' : 'badge-info'
                    }">
                        ${layerAnalysis.biased_layer.sensitivity > 0.5 ? 'High Sensitivity' : 'Moderate'}
                    </span>
                </div>
                <div class="card-value">${layerAnalysis.biased_layer.layer_name}</div>
                <div class="card-description">
                    Contains ${
                        layerAnalysis.biased_layer.neuron_count
                    } neurons with sensitivity score of ${layerAnalysis.biased_layer.sensitivity.toFixed(4)}
                </div>
            </div>
        </div>

        <div class="chart-container">
            <div class="chart-title">Layer-Wise Bias Sensitivity</div>
            <div id="layer-chart" style="height: 300px;"></div>
        </div>
    </div>

    <!-- Neuron Analysis Section -->
    <div class="section">
        <div class="section-header">
            <div class="section-icon" style="background: rgba(244, 67, 54, 0.2);">
                <span style="color: var(--accent-red);">⚡</span>
            </div>
            <h2 class="section-title">Neuron-Level Localization</h2>
        </div>

        <div class="cards-grid">
            <div class="card" style="grid-column: span 2;">
                <div class="card-header">
                    <span class="card-label">Top Biased Neurons</span>
                </div>
                <div class="neuron-list">
                    ${neuronAnalysis
                        .map(
                            (n: any, idx: number) => `
                        <div class="neuron-item">
                            <div class="neuron-info">
                                <div class="neuron-rank">${idx + 1}</div>
                                <span class="neuron-name">Neuron ${n.neuron_idx}</span>
                            </div>
                            <span class="neuron-score">Impact: ${n.impact_score.toFixed(4)}</span>
                        </div>
                    `,
                        )
                        .join('')}
                </div>
            </div>
        </div>

        <div class="chart-container">
            <div class="chart-title">Neuron Impact Scores</div>
            <div id="neuron-chart" style="height: 300px;"></div>
        </div>
    </div>
    `
            : ''
    }

    ${
        explanationsData?.shap
            ? `
    <!-- SHAP Explanations Section -->
    <div class="section">
        <div class="section-header">
            <div class="section-icon" style="background: rgba(255, 152, 0, 0.2);">
                <span style="color: var(--accent-orange);">📊</span>
            </div>
            <h2 class="section-title">SHAP Feature Importance</h2>
        </div>
        <div class="interpretation-box" style="margin-bottom: 16px;">
            <div class="interpretation-title">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1zm0 12.5a5.5 5.5 0 1 1 0-11 5.5 5.5 0 0 1 0 11zM7 4h2v5H7V4zm0 6h2v2H7v-2z"/></svg>
                What are SHAP Values?
            </div>
            <div class="interpretation-content">
                <strong>SHAP (SHapley Additive exPlanations)</strong> uses game theory to compute the contribution of each feature to the model's prediction.
                Higher bars indicate features with more influence. Features related to protected attributes may indicate bias pathways.
            </div>
        </div>
        <div class="chart-container">
            <div class="chart-title">Global Feature Importance (Mean |SHAP Value|)</div>
            <div id="shap-global-chart" style="height: 400px;"></div>
        </div>
        <div class="chart-container" style="margin-top: 16px;">
            <div class="chart-title">SHAP Values for Individual Instances</div>
            <div id="shap-beeswarm-chart" style="height: 400px;"></div>
        </div>
    </div>
    `
            : ''
    }

    ${
        explanationsData?.lime
            ? `
    <!-- LIME Explanations Section -->
    <div class="section">
        <div class="section-header">
            <div class="section-icon" style="background: rgba(79, 195, 247, 0.2);">
                <span style="color: var(--accent-blue);">🔍</span>
            </div>
            <h2 class="section-title">LIME Local Explanations</h2>
        </div>
        <div class="interpretation-box" style="margin-bottom: 16px;">
            <div class="interpretation-title">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1zm0 12.5a5.5 5.5 0 1 1 0-11 5.5 5.5 0 0 1 0 11zM7 4h2v5H7V4zm0 6h2v2H7v-2z"/></svg>
                What is LIME?
            </div>
            <div class="interpretation-content">
                <strong>LIME (Local Interpretable Model-agnostic Explanations)</strong> explains individual predictions by learning a simple local model.
                It shows which features push the prediction toward or away from each class. Protected attributes appearing prominently suggest discrimination.
            </div>
        </div>
        <div class="chart-container">
            <div class="chart-title">Aggregated Feature Importance (LIME)</div>
            <div id="lime-global-chart" style="height: 400px;"></div>
        </div>
    </div>
    `
            : ''
    }

    <!-- Interpretation Section -->
    <div class="section">
        <div class="section-header">
            <div class="section-icon" style="background: rgba(79, 195, 247, 0.2);">
                <span style="color: var(--accent-blue);">💡</span>
            </div>
            <h2 class="section-title">Understanding the Results</h2>
        </div>

        <div class="interpretation-box">
            <div class="interpretation-title">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1zm0 12.5a5.5 5.5 0 1 1 0-11 5.5 5.5 0 0 1 0 11zM7 4h2v5H7V4zm0 6h2v2H7v-2z"/></svg>
                What is QID (Quantitative Individual Discrimination)?
            </div>
            <div class="interpretation-content">
                QID measures how many <strong>bits of protected information</strong> (e.g., gender, race, age) the model uses to make its predictions.
                A value of <strong>0 bits</strong> means the model is perfectly fair and doesn't use any protected attributes.
                <strong>Higher values indicate more bias</strong> - the model is learning to discriminate based on protected characteristics.
            </div>
        </div>

        <div class="interpretation-box" style="margin-top: 16px;">
            <div class="interpretation-title">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1zm0 12.5a5.5 5.5 0 1 1 0-11 5.5 5.5 0 0 1 0 11zM7 4h2v5H7V4zm0 6h2v2H7v-2z"/></svg>
                The 80% Rule (Four-Fifths Rule)
            </div>
            <div class="interpretation-content">
                The <strong>disparate impact ratio</strong> should be ≥ <strong>0.8 (80%)</strong> to comply with legal standards in employment and lending.
                This means the selection rate for a protected group should be at least 80% of the rate for the most favored group.
                <strong>Values below 0.8 may indicate legally actionable discrimination</strong> and could expose your organization to regulatory scrutiny.
            </div>
        </div>
    </div>

    <script>
        // Chart styling
        const chartLayout = {
            plot_bgcolor: '#252526',
            paper_bgcolor: '#252526',
            font: { color: '#e4e4e4', family: '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif' },
            margin: { t: 20, r: 20, b: 50, l: 60 },
            xaxis: { gridcolor: '#3c3c3c', zerolinecolor: '#3c3c3c' },
            yaxis: { gridcolor: '#3c3c3c', zerolinecolor: '#3c3c3c' }
        };

        // QID values for charts
        const instances = ${JSON.stringify(searchResults.discriminatory_instances)};
        const qidValues = instances.map(inst => inst.qid);
        const disparateImpact = ${qidMetrics.mean_disparate_impact};

        // 1. QID Histogram
        if (qidValues.length > 0) {
            const histogramTrace = {
                x: qidValues,
                type: 'histogram',
                nbinsx: 20,
                marker: {
                    color: '#ff9800',
                    line: { color: '#fff', width: 1 }
                },
                opacity: 0.85,
                hovertemplate: 'QID Range: %{x}<br>Count: %{y}<extra></extra>'
            };

            Plotly.newPlot('qid-histogram', [histogramTrace], {
                ...chartLayout,
                margin: { t: 10, r: 20, b: 50, l: 50 },
                xaxis: {
                    ...chartLayout.xaxis,
                    title: { text: 'QID (bits)', font: { color: '#a0a0a0' } }
                },
                yaxis: {
                    ...chartLayout.yaxis,
                    title: { text: 'Frequency', font: { color: '#a0a0a0' } }
                },
                bargap: 0.05
            }, { responsive: true });
        }

        // 2. Disparate Impact Gauge
        const gaugeValue = disparateImpact * 100;
        const gaugeColor = disparateImpact >= 0.8 ? '#4caf50' : (disparateImpact >= 0.6 ? '#ff9800' : '#f44336');

        const gaugeTrace = {
            type: 'indicator',
            mode: 'gauge+number+delta',
            value: gaugeValue,
            title: {
                text: 'Disparate Impact Ratio',
                font: { size: 16, color: '#e4e4e4' }
            },
            number: {
                suffix: '%',
                font: { size: 36, color: '#e4e4e4' }
            },
            delta: {
                reference: 80,
                increasing: { color: '#4caf50' },
                decreasing: { color: '#f44336' },
                font: { size: 14 }
            },
            gauge: {
                axis: {
                    range: [0, 100],
                    tickwidth: 1,
                    tickcolor: '#3c3c3c',
                    tickfont: { color: '#a0a0a0' }
                },
                bar: { color: gaugeColor, thickness: 0.75 },
                bgcolor: '#1e1e1e',
                borderwidth: 2,
                bordercolor: '#3c3c3c',
                steps: [
                    { range: [0, 60], color: 'rgba(244, 67, 54, 0.3)' },
                    { range: [60, 80], color: 'rgba(255, 152, 0, 0.3)' },
                    { range: [80, 100], color: 'rgba(76, 175, 80, 0.3)' }
                ],
                threshold: {
                    line: { color: '#ff5252', width: 4 },
                    thickness: 0.75,
                    value: 80
                }
            }
        };

        Plotly.newPlot('disparate-impact-gauge', [gaugeTrace], {
            ...chartLayout,
            margin: { t: 50, r: 30, b: 30, l: 30 }
        }, { responsive: true });

        // 3. QID Scatter Plot (original chart)
        if (instances.length > 0) {
            const trace = {
                x: instances.map((_, i) => i + 1),
                y: instances.map(inst => inst.qid),
                type: 'scatter',
                mode: 'markers',
                marker: {
                    size: 10,
                    color: instances.map(inst => inst.qid),
                    colorscale: [
                        [0, '#4caf50'],
                        [0.5, '#ff9800'],
                        [1, '#f44336']
                    ],
                    showscale: true,
                    colorbar: {
                        title: { text: 'QID (bits)', font: { color: '#e4e4e4' } },
                        tickfont: { color: '#a0a0a0' }
                    }
                },
                text: instances.map(inst => \`QID: \${inst.qid.toFixed(4)} bits<br>Variance: \${inst.variance.toFixed(4)}\`),
                hoverinfo: 'text'
            };

            Plotly.newPlot('qid-chart', [trace], {
                ...chartLayout,
                xaxis: { ...chartLayout.xaxis, title: { text: 'Instance Index', font: { color: '#a0a0a0' } } },
                yaxis: { ...chartLayout.yaxis, title: { text: 'QID (bits)', font: { color: '#a0a0a0' } } }
            }, { responsive: true });
        }

        // Layer Sensitivity Chart
        const layerData = ${JSON.stringify(layerAnalysis?.all_layers || [])};
        if (layerData.length > 0) {
            const layerTrace = {
                x: layerData.map(l => l.layer_name),
                y: layerData.map(l => l.sensitivity),
                type: 'bar',
                marker: {
                    color: layerData.map(l => l.sensitivity),
                    colorscale: [
                        [0, '#4fc3f7'],
                        [0.5, '#9c27b0'],
                        [1, '#f44336']
                    ],
                    line: { width: 0 }
                },
                hovertemplate: '%{x}<br>Sensitivity: %{y:.4f}<extra></extra>'
            };

            Plotly.newPlot('layer-chart', [layerTrace], {
                ...chartLayout,
                xaxis: { ...chartLayout.xaxis, title: { text: 'Layer', font: { color: '#a0a0a0' } } },
                yaxis: { ...chartLayout.yaxis, title: { text: 'Sensitivity Score', font: { color: '#a0a0a0' } } }
            }, { responsive: true });
        }

        // Neuron Impact Chart
        const neuronData = ${JSON.stringify(neuronAnalysis || [])};
        if (neuronData.length > 0) {
            const neuronTrace = {
                x: neuronData.map(n => 'N' + n.neuron_idx),
                y: neuronData.map(n => n.impact_score),
                type: 'bar',
                marker: {
                    color: '#f44336',
                    line: { width: 0 }
                },
                hovertemplate: 'Neuron %{x}<br>Impact: %{y:.4f}<extra></extra>'
            };

            Plotly.newPlot('neuron-chart', [neuronTrace], {
                ...chartLayout,
                xaxis: { ...chartLayout.xaxis, title: { text: 'Neuron', font: { color: '#a0a0a0' } } },
                yaxis: { ...chartLayout.yaxis, title: { text: 'Impact Score', font: { color: '#a0a0a0' } } }
            }, { responsive: true });
        }

        // Internal Space Activation Charts
        const activationsData = ${JSON.stringify(activationsData)};
        if (activationsData && activationsData.layers) {
            activationsData.layers.forEach(function(layer, idx) {
                const traceByLabel = {
                    x: layer.x,
                    y: layer.y,
                    mode: 'markers',
                    type: 'scatter',
                    marker: {
                        size: 5,
                        color: activationsData.labels,
                        colorscale: [[0, '#4fc3f7'], [1, '#f44336']],
                        opacity: 0.6,
                        showscale: true,
                        colorbar: { title: { text: 'Label', font: { color: '#a0a0a0' } }, tickfont: { color: '#a0a0a0' } }
                    },
                    text: activationsData.labels.map(function(l, i) {
                        return 'Label: ' + l + '<br>Protected: ' + activationsData.protected[i].toFixed(2);
                    }),
                    hoverinfo: 'text'
                };
                Plotly.newPlot('activation-layer-' + idx, [traceByLabel], {
                    ...chartLayout,
                    xaxis: { ...chartLayout.xaxis, title: { text: 'Component 1', font: { color: '#a0a0a0' } } },
                    yaxis: { ...chartLayout.yaxis, title: { text: 'Component 2', font: { color: '#a0a0a0' } } }
                }, { responsive: true });
            });
        }

        // SHAP Charts
        const shapData = ${JSON.stringify(explanationsData?.shap || null)};
        if (shapData) {
            // Global importance bar chart (sorted)
            const shapIndices = shapData.global_importance
                .map(function(val, idx) { return { val: val, idx: idx }; })
                .sort(function(a, b) { return b.val - a.val; });

            Plotly.newPlot('shap-global-chart', [{
                y: shapIndices.map(function(d) { return shapData.feature_names[d.idx]; }),
                x: shapIndices.map(function(d) { return d.val; }),
                type: 'bar',
                orientation: 'h',
                marker: { color: '#ff9800', line: { width: 0 } },
                hovertemplate: '%{y}: %{x:.4f}<extra></extra>'
            }], {
                ...chartLayout,
                margin: { t: 20, r: 20, b: 50, l: 150 },
                yaxis: { ...chartLayout.yaxis, autorange: 'reversed' },
                xaxis: { ...chartLayout.xaxis, title: { text: 'Mean |SHAP Value|', font: { color: '#a0a0a0' } } }
            }, { responsive: true });

            // Beeswarm chart (SHAP values per feature for each instance)
            if (shapData.shap_values && shapData.shap_values.length > 0) {
                const beeswarmTraces = [];
                for (var instIdx = 0; instIdx < shapData.shap_values.length; instIdx++) {
                    beeswarmTraces.push({
                        y: shapData.feature_names,
                        x: shapData.shap_values[instIdx],
                        type: 'scatter',
                        mode: 'markers',
                        marker: { size: 6, opacity: 0.6 },
                        name: 'Instance ' + (instIdx + 1),
                        hovertemplate: '%{y}: %{x:.4f}<extra></extra>'
                    });
                }
                Plotly.newPlot('shap-beeswarm-chart', beeswarmTraces, {
                    ...chartLayout,
                    margin: { t: 20, r: 20, b: 50, l: 150 },
                    showlegend: false,
                    xaxis: { ...chartLayout.xaxis, title: { text: 'SHAP Value (impact on prediction)', font: { color: '#a0a0a0' } } }
                }, { responsive: true });
            }
        }

        // LIME Charts
        const limeData = ${JSON.stringify(explanationsData?.lime || null)};
        if (limeData) {
            const limeIndices = limeData.aggregated_importance
                .map(function(val, idx) { return { val: val, idx: idx }; })
                .sort(function(a, b) { return b.val - a.val; });

            Plotly.newPlot('lime-global-chart', [{
                y: limeIndices.map(function(d) { return limeData.feature_names[d.idx]; }),
                x: limeIndices.map(function(d) { return d.val; }),
                type: 'bar',
                orientation: 'h',
                marker: { color: '#4fc3f7', line: { width: 0 } },
                hovertemplate: '%{y}: %{x:.4f}<extra></extra>'
            }], {
                ...chartLayout,
                margin: { t: 20, r: 20, b: 50, l: 150 },
                yaxis: { ...chartLayout.yaxis, autorange: 'reversed' },
                xaxis: { ...chartLayout.xaxis, title: { text: 'Mean |LIME Weight|', font: { color: '#a0a0a0' } } }
            }, { responsive: true });
        }
    </script>
</body>
</html>
    `;
}

function calculateFairnessScore(qidMetrics: any): number {
    // Calculate a 0-100 fairness score based on multiple metrics
    let score = 100;

    // Penalize for high mean QID (0-2 bits range maps to 0-30 penalty)
    score -= Math.min(qidMetrics.mean_qid * 15, 30);

    // Penalize for discriminatory instances (0-100% maps to 0-30 penalty)
    score -= Math.min(qidMetrics.pct_discriminatory * 0.3, 30);

    // Penalize for low disparate impact (0.8-1.0 is good, below 0.8 is bad)
    if (qidMetrics.mean_disparate_impact < 0.8) {
        score -= (0.8 - qidMetrics.mean_disparate_impact) * 50;
    }

    // Penalize for high max QID
    score -= Math.min(qidMetrics.max_qid * 5, 20);

    return Math.max(0, Math.round(score));
}

function getFairnessStatus(score: number): { label: string; class: string; color: string } {
    if (score >= 80) {
        return { label: 'Good', class: 'status-good', color: '#4caf50' };
    } else if (score >= 60) {
        return { label: 'Needs Review', class: 'status-warning', color: '#ff9800' };
    } else {
        return { label: 'Concerning', class: 'status-danger', color: '#f44336' };
    }
}

export function deactivate() {
    if (serverProcess) {
        serverProcess.kill();
        console.log('Backend server stopped');
    }
}
