document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const submitBtn = document.getElementById('submit-btn');
    const fileDropArea = document.querySelector('.file-drop-area');
    const filePreviewWrapper = document.getElementById('file-preview-wrapper');
    const filePreviewList = document.getElementById('file-preview-list');
    const loader = document.getElementById('results-loader');
    const resultsSection = document.getElementById('results-section');

    const API_BASE_URL = 'http://localhost:8000';

    function handleFiles(files) {
        if (files.length === 0) {
            filePreviewWrapper.classList.add('hidden');
            submitBtn.disabled = true;
            return;
        }

        filePreviewList.innerHTML = '';
        Array.from(files).forEach(file => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-preview-item';
            fileItem.textContent = file.name;
            filePreviewList.appendChild(fileItem);
        });

        filePreviewWrapper.classList.remove('hidden');
        submitBtn.disabled = false;
        submitBtn.textContent = `Process ${files.length} File(s)`;
    }

    fileInput.addEventListener('change', () => handleFiles(fileInput.files));
    fileDropArea.addEventListener('dragover', (e) => { e.preventDefault(); fileDropArea.classList.add('is-active'); });
    fileDropArea.addEventListener('dragleave', () => fileDropArea.classList.remove('is-active'));
    fileDropArea.addEventListener('drop', (e) => {
        e.preventDefault();
        fileDropArea.classList.remove('is-active');
        fileInput.files = e.dataTransfer.files;
        handleFiles(fileInput.files);
    });

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const files = fileInput.files;
        if (files.length === 0) return;

        // UI updates for loading
        submitBtn.disabled = true;
        loader.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        resultsSection.innerHTML = '';

        try {
            if (files.length === 1) {
                // --- SINGLE FILE PROCESSING ---
                const formData = new FormData();
                formData.append('file', files[0]);
                const response = await fetch(`${API_BASE_URL}/process`, { method: 'POST', body: formData });
                if (!response.ok) throw new Error((await response.json()).detail || 'Server error');
                const result = await response.json();
                displaySingleResult(result);
            } else {
                // --- BATCH FILE PROCESSING ---
                const formData = new FormData();
                Array.from(files).forEach(file => formData.append('files', file));
                const response = await fetch(`${API_BASE_URL}/batch`, { method: 'POST', body: formData });
                if (!response.ok) throw new Error((await response.json()).detail || 'Server error');
                const batchResult = await response.json();
                displayBatchResults(batchResult);
            }
        } catch (error) {
            displayError(error.message);
        } finally {
            loader.classList.add('hidden');
            resultsSection.classList.remove('hidden');
            submitBtn.disabled = false;
            submitBtn.textContent = `Process File(s)`;
        }
    });

    function displaySingleResult(result) {
        const resultHtml = createResultCardHtml(result);
        resultsSection.innerHTML = resultHtml;
        addCardToggleListeners();
    }

    function displayBatchResults(batchData) {
        const hasErrors = batchData.failed > 0;
        const summaryClass = hasErrors ? 'has-errors' : '';
        const summaryHtml = `
            <div class="result-batch-summary ${summaryClass}">
                Batch Complete: ${batchData.successful} successful, ${batchData.failed} failed in ${batchData.total_time.toFixed(2)}s
            </div>
        `;

        const resultsHtml = batchData.results.map(createResultCardHtml).join('');
        resultsSection.innerHTML = summaryHtml + resultsHtml;
        addCardToggleListeners();
    }
    
    function createResultCardHtml(result) {
        const isError = result.errors.length > 0;
        const statusClass = isError ? 'error' : 'success';
        const statusText = isError ? 'Failed' : 'Success';

        // Group data as requested
        const personalInfo = { ...result.personal_info };
        const documentInfo = {
            "Document Type": result.mrz_data?.document_type,
            ...result.document_info
        };

        return `
            <div class="result-card ${isError ? '' : 'is-open'}">
                <header class="result-card-header">
                    <span class="filename">${result.filename}</span>
                    <span class="status ${statusClass}">${statusText}</span>
                </header>
                <div class="result-card-body">
                    ${isError ? `<div class="info-item error-text"><strong>Error:</strong> <span>${result.errors.join(', ')}</span></div>` : ''}
                    <div class="info-grid">
                        <div class="info-box">${createInfoBoxHtml('👤 Personal Information', personalInfo)}</div>
                        <div class="info-box">${createInfoBoxHtml('💳 Document Information', documentInfo)}</div>
                    </div>
                    <details class="raw-data-toggle">
                        <summary>Show Raw Data</summary>
                        <pre>${JSON.stringify(result, null, 2)}</pre>
                    </details>
                </div>
            </div>
        `;
    }

    function createInfoBoxHtml(title, data) {
        const itemsHtml = Object.entries(data)
            .map(([key, value]) => {
                if (!value) return ''; // Don't show empty fields
                const formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                return `
                    <div class="info-item">
                        <strong>${formattedKey}</strong>
                        <span>${value}</span>
                    </div>`;
            }).join('');

        return `<h4>${title}</h4>${itemsHtml}`;
    }

    function displayError(message) {
        resultsSection.innerHTML = `<div class="result-wrapper" style="text-align: center; color: var(--error-text);"><strong>An error occurred:</strong> ${message}</div>`;
    }

    function addCardToggleListeners() {
        document.querySelectorAll('.result-card-header').forEach(header => {
            header.addEventListener('click', () => {
                header.parentElement.classList.toggle('is-open');
            });
        });
    }
});