import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { getConfig } from './config/settings';
import { startBackend, stopBackend } from './server/backend';
import { initStatusBar } from './server/statusBar';
import { showError } from './server/errors';
import { fetchColumns, ColumnData } from './analysis/columns';
import { runAnalysis } from './analysis/pipeline';

export async function activate(context: vscode.ExtensionContext): Promise<void> {
    console.log('Fairness DL Extension activating...');

    initStatusBar(context);
    await startBackend(context);

    context.subscriptions.push(
        vscode.commands.registerCommand('fairlint-dl.analyzeDataset', analyzeDataset),
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('fairlint-dl.analyzeFromMenu', async (uri: vscode.Uri) => {
            await analyzeDataset(uri);
        }),
    );
}

async function analyzeDataset(uri: vscode.Uri): Promise<void> {
    const filePath = uri.fsPath;

    if (!filePath.endsWith('.csv')) {
        showError('Invalid File Type', 'Please select a CSV file (.csv extension required).');
        return;
    }

    if (!fs.existsSync(filePath)) {
        showError('File Not Found', `The file "${path.basename(filePath)}" does not exist.`);
        return;
    }

    // Step 1: Fetch columns
    const columnData: ColumnData | null = await vscode.window.withProgress(
        { location: vscode.ProgressLocation.Notification, title: 'Loading CSV columns...', cancellable: false },
        async () => fetchColumns(filePath),
    );

    if (!columnData || columnData.columns.length === 0) {
        showError('Empty Dataset', 'The CSV file appears to be empty or has no columns.');
        return;
    }

    // Step 2: Select label column
    const quickPickItems: vscode.QuickPickItem[] = columnData.columns.map((col: string, index: number) => {
        const sampleValue = columnData.sampleData[0]?.[col];
        const sampleStr = sampleValue !== undefined ? ` (e.g., "${sampleValue}")` : '';
        return {
            label: col,
            description: `Column ${index + 1}${sampleStr}`,
            detail: `Select this column as the target/label variable`,
        };
    });

    const selectedColumn = await vscode.window.showQuickPick(quickPickItems, {
        placeHolder: 'Select the target/label column for prediction',
        title: 'Fairness Analysis: Select Label Column',
        matchOnDescription: true,
        matchOnDetail: false,
    });

    if (!selectedColumn) {
        return;
    }

    const labelColumn = selectedColumn.label;

    // Step 3: Select protected attributes
    const protectedItems: vscode.QuickPickItem[] = columnData.columns
        .filter((col: string) => col !== labelColumn)
        .map((col: string) => ({
            label: col,
            picked: columnData.detectedSensitive.includes(col),
            description: columnData.detectedSensitive.includes(col) ? '(auto-detected)' : '',
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

    // Step 4: Select DNN architecture
    const defaultLayers = getConfig().hiddenLayers;
    const archChoice = await vscode.window.showQuickPick(
        [
            {
                label: `Default (${defaultLayers})`,
                description: 'DICE paper architecture',
                detail: `Hidden layers: [${defaultLayers}] -> Output(2)`,
                value: defaultLayers,
            },
            {
                label: 'Wide (128,64,32,16)',
                description: '4-layer wider network',
                detail: 'Hidden layers: [128, 64, 32, 16] -> Output(2)',
                value: '128,64,32,16',
            },
            {
                label: 'Deep (256,128,64,32,16,8)',
                description: '6-layer deep network',
                detail: 'Hidden layers: [256, 128, 64, 32, 16, 8] -> Output(2)',
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
        return;
    }

    let hiddenLayersStr = (archChoice as unknown as { value: string }).value;

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
            return;
        }
        hiddenLayersStr = customInput;
    }

    const hiddenLayers = hiddenLayersStr
        .split(',')
        .map((s: string) => parseInt(s.trim(), 10))
        .filter((n: number) => !isNaN(n) && n > 0);

    await runAnalysis(filePath, labelColumn, protectedFeatures, hiddenLayers);
}

export function deactivate(): void {
    stopBackend();
}
