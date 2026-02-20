import * as vscode from 'vscode';
import { getConfig } from '../config/settings';

let statusBarItem: vscode.StatusBarItem;

export function initStatusBar(context: vscode.ExtensionContext): void {
    statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
    statusBarItem.text = '$(pulse) FairLint-DL';
    statusBarItem.tooltip = 'Click to analyze a dataset for fairness';
    statusBarItem.command = 'fairlint-dl.analyzeDataset';
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);
}

export function updateStatusBar(text: string, tooltip?: string, color?: string): void {
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

export function resetStatusBar(): void {
    statusBarItem.text = '$(pulse) FairLint-DL';
    statusBarItem.tooltip = 'Click to analyze a dataset for fairness';
    statusBarItem.backgroundColor = undefined;
    statusBarItem.show();
}

export function setStatusBarSuccess(message: string): void {
    updateStatusBar(`$(check) ${message}`, 'Analysis complete - click to analyze another dataset');
    setTimeout(resetStatusBar, 5000);
}

export function setStatusBarError(message: string): void {
    updateStatusBar(`$(error) ${message}`, 'Analysis failed - click to try again', 'statusBarItem.errorBackground');
    setTimeout(resetStatusBar, 8000);
}
