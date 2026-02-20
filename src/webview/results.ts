import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { getWebviewHtml } from './htmlBuilder';

export function showResults(results: {
    training: { accuracy: number; protected_features: string[]; num_parameters: number };
    analysis: { qid_metrics: Record<string, number> };
    search: { search_results: { discriminatory_instances: unknown[]; best_qid: number; num_found: number } };
    debug?: { layer_analysis: Record<string, unknown>; neuron_analysis: unknown[] };
    activations?: Record<string, unknown>;
    explanations?: { shap?: Record<string, unknown>; lime?: Record<string, unknown> };
    metadata?: { file?: string; filePath?: string; labelColumn?: string; totalTime?: number };
    config?: Record<string, unknown>;
}): void {
    const panel = vscode.window.createWebviewPanel(
        'fairnessResults',
        'Fairness Analysis Results',
        vscode.ViewColumn.One,
        { enableScripts: true, retainContextWhenHidden: true },
    );

    panel.webview.html = getWebviewHtml(results);

    panel.webview.onDidReceiveMessage(async (message) => {
        if (message.command === 'saveJson') {
            await handleSaveJson(results);
        }
    });
}

async function handleSaveJson(results: {
    training: { accuracy: number; protected_features: string[]; num_parameters: number };
    analysis: { qid_metrics: Record<string, number> };
    search: { search_results: { best_qid: number; num_found: number } };
    debug?: { layer_analysis: Record<string, unknown>; neuron_analysis: unknown[] };
    explanations?: { shap?: Record<string, unknown>; lime?: Record<string, unknown> };
    metadata?: { file?: string; filePath?: string };
}): Promise<void> {
    const baseFileName = results.metadata?.file?.replace('.csv', '') || 'fairness-report';
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const defaultDir = results.metadata?.filePath
        ? path.dirname(results.metadata.filePath)
        : vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '';
    const defaultName = `${baseFileName}_data_${timestamp}.json`;
    const defaultUri = defaultDir ? vscode.Uri.file(path.join(defaultDir, defaultName)) : undefined;

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
                    shap_global_importance: (results.explanations?.shap as Record<string, unknown>)?.global_importance,
                    shap_feature_names: (results.explanations?.shap as Record<string, unknown>)?.feature_names,
                    lime_aggregated_importance: (results.explanations?.lime as Record<string, unknown>)?.aggregated_importance,
                    lime_feature_names: (results.explanations?.lime as Record<string, unknown>)?.feature_names,
                },
            };
            fs.writeFileSync(uri.fsPath, JSON.stringify(exportData, null, 2), 'utf-8');
            vscode.window.showInformationMessage(`Data exported to ${path.basename(uri.fsPath)}`);
        } catch (err: unknown) {
            const error = err as Error;
            vscode.window.showErrorMessage(`Failed to export data: ${error.message}`);
        }
    }
}
