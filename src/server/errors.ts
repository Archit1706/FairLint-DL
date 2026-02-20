import * as vscode from 'vscode';
import axios, { AxiosError } from 'axios';

const ERROR_MESSAGES: Record<string, string> = {
    ECONNREFUSED: 'Cannot connect to the analysis server. Please wait for it to start or restart VS Code.',
    ETIMEDOUT: 'The analysis is taking longer than expected. This may happen with large datasets.',
    ENOTFOUND: 'Network error. Please check your connection.',
    'Train model first': 'Please run training first before performing analysis.',
};

export function showError(
    title: string,
    detail: string,
    actions?: { label: string; action: () => void }[],
): void {
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

export function parseError(error: unknown): { title: string; detail: string; suggestion: string } {
    if (axios.isAxiosError(error)) {
        const axiosError = error as AxiosError;

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

        if (axiosError.response) {
            const status = axiosError.response.status;
            const data = axiosError.response.data as Record<string, string>;
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

        if (axiosError.code === 'ECONNABORTED') {
            return {
                title: 'Timeout',
                detail: 'The operation took too long to complete.',
                suggestion: 'Try with a smaller dataset or fewer epochs.',
            };
        }
    }

    const err = error as Error;
    return {
        title: 'Error',
        detail: err.message || 'An unknown error occurred.',
        suggestion: 'Please try again or check the console for details.',
    };
}
