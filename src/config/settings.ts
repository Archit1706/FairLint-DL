import * as vscode from 'vscode';

export interface FairLintConfig {
    epochs: number;
    batchSize: number;
    hiddenLayers: string;
    qidThreshold: number;
    maxSamples: number;
    globalIterations: number;
    localNeighbors: number;
    serverPort: number;
    autoDetectProtected: boolean;
    showCharts: boolean;
    theme: string;
    showProgress: boolean;
    showStatusBar: boolean;
}

export function getConfig(): FairLintConfig {
    const config = vscode.workspace.getConfiguration('fairlint-dl');
    return {
        epochs: config.get<number>('training.epochs', 30),
        batchSize: config.get<number>('training.batchSize', 32),
        hiddenLayers: config.get<string>('training.hiddenLayers', '64,32,16,8,4'),
        qidThreshold: config.get<number>('analysis.qidThreshold', 0.1),
        maxSamples: config.get<number>('analysis.maxSamples', 500),
        globalIterations: config.get<number>('search.globalIterations', 50),
        localNeighbors: config.get<number>('search.localNeighbors', 30),
        serverPort: config.get<number>('server.port', 8765),
        autoDetectProtected: config.get<boolean>('detection.autoDetectProtected', true),
        showCharts: config.get<boolean>('visualization.showCharts', true),
        theme: config.get<string>('visualization.theme', 'dark'),
        showProgress: config.get<boolean>('notifications.showProgress', true),
        showStatusBar: config.get<boolean>('notifications.showStatusBar', true),
    };
}

export function getServerUrl(): string {
    return `http://localhost:${getConfig().serverPort}`;
}
