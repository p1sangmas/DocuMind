// DocuMind Web Interface JavaScript

// Global Variables
const API_BASE_URL = 'http://localhost:8080/api';
let sessionId = null;
let conversationHistory = [];
let processedFiles = [];
let selectedFiles = []; // Array to track selected files

// DOM Elements
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const chatMessages = document.getElementById('chat-messages');
const chatContainer = document.getElementById('chat-container');
const noDocsWarning = document.getElementById('no-docs-warning');
const fileUpload = document.getElementById('file-upload');
const processDocsBtn = document.getElementById('process-docs-btn');
const processedFilesList = document.getElementById('processed-files-list');
const docCount = document.getElementById('doc-count');
const analyticsDocCount = document.getElementById('analytics-doc-count');
const embedModel = document.getElementById('embed-model');
const clearKbBtn = document.getElementById('clear-kb-btn');
const confirmClear = document.getElementById('confirm-clear');
const confirmClearCheckbox = document.getElementById('confirm-clear-checkbox');
const confirmClearBtn = document.getElementById('confirm-clear-btn');
const ollamaStatus = document.getElementById('ollama-status');
const ocrStatus = document.getElementById('ocr-status');
const ollamaWarning = document.getElementById('ollama-warning');
const ocrWarning = document.getElementById('ocr-warning');
const ollamaHelpBtn = document.getElementById('ollama-help-btn');
const ocrHelpBtn = document.getElementById('ocr-help-btn');
const modalOverlay = document.getElementById('modal-overlay');
const ocaHelpModal = document.getElementById('ocr-help-modal');
const ollamaHelpModal = document.getElementById('ollama-help-modal');
const closeModalBtns = document.querySelectorAll('.close-modal-btn');
const sessionIdEl = document.getElementById('session-id');
const conversationCountEl = document.getElementById('conversation-count');
const feedbackAnalysis = document.getElementById('feedback-analysis');
const tabBtns = document.querySelectorAll('.tab-btn');
const tabPanes = document.querySelectorAll('.tab-pane');
const exportConversationBtn = document.getElementById('export-conversation-btn');
const clearConversationBtn = document.getElementById('clear-conversation-btn');
const systemInfo = document.getElementById('system-info');
const expandBtns = document.querySelectorAll('.expand-btn');

// Initialize the application
function init() {
    // Generate a session ID
    sessionId = generateSessionId();
    sessionIdEl.textContent = sessionId;
    
    // Check system status
    checkSystemStatus();
    
    // Update processed files list
    updateProcessedFilesList();
    
    // Set up event listeners
    setupEventListeners();
}

// Generate a unique session ID
function generateSessionId() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

// Check system status
async function checkSystemStatus() {
    try {
        // Set loading indicators
        ollamaStatus.className = 'status-indicator';
        ollamaStatus.classList.add('loading');
        ocrStatus.className = 'status-indicator';
        ocrStatus.classList.add('loading');
        docCount.textContent = "...";
        
        const response = await fetch(`${API_BASE_URL}/status`);
        const data = await response.json();
        
        // Update system status indicators
        updateStatusIndicator(ollamaStatus, data.ollama_available);
        updateStatusIndicator(ocrStatus, data.ocr_available);
        
        // Show/hide warnings
        ollamaWarning.classList.toggle('hidden', data.ollama_available);
        ocrWarning.classList.toggle('hidden', data.ocr_available);
        
        // Update collection info
        updateCollectionInfo(data.collection);
        
        // Update chat container visibility based on document count
        updateChatVisibility(data.collection?.document_count || 0);
        
    } catch (error) {
        console.error('Error checking system status:', error);
        showToast('Error connecting to API server. Please make sure the server is running.', 'error');
    }
}

// Update status indicator
function updateStatusIndicator(element, isAvailable) {
    element.className = 'status-indicator';
    element.classList.add(isAvailable ? 'online' : 'offline');
}

// Update collection info
function updateCollectionInfo(collection) {
    if (collection) {
        docCount.textContent = collection.document_count || 0;
        analyticsDocCount.textContent = collection.document_count || 0;
        
        if (collection.embedding_model) {
            embedModel.textContent = collection.embedding_model;
        }
    }
}

// Update chat visibility based on document count
function updateChatVisibility(documentCount) {
    if (documentCount > 0) {
        chatContainer.classList.remove('hidden');
        noDocsWarning.classList.add('hidden');
    } else {
        chatContainer.classList.add('hidden');
        noDocsWarning.classList.remove('hidden');
    }
}

// Set up event listeners
function setupEventListeners() {
    // Chat input
    chatInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Send button
    sendBtn.addEventListener('click', sendMessage);
    
    // File upload
    fileUpload.addEventListener('change', handleFileUpload);
    
    // Process documents button
    processDocsBtn.addEventListener('click', processDocuments);
    
    // Clear knowledge base
    clearKbBtn.addEventListener('click', () => {
        confirmClear.classList.toggle('hidden');
    });
    
    // Clear knowledge base confirmation checkbox
    confirmClearCheckbox.addEventListener('change', () => {
        confirmClearBtn.disabled = !confirmClearCheckbox.checked;
    });
    
    // Confirm clear knowledge base
    confirmClearBtn.addEventListener('click', clearKnowledgeBase);
    
    // Help buttons
    ollamaHelpBtn.addEventListener('click', () => {
        showModal(ollamaHelpModal);
    });
    
    ocrHelpBtn.addEventListener('click', () => {
        showModal(ocaHelpModal);
    });
    
    // Close modal buttons
    closeModalBtns.forEach(button => {
        button.addEventListener('click', closeModals);
    });
    
    // Click outside modal to close
    modalOverlay.addEventListener('click', function(e) {
        if (e.target === modalOverlay) {
            closeModals();
        }
    });
    
    // Tab buttons
    tabBtns.forEach(button => {
        button.addEventListener('click', () => {
            switchTab(button.dataset.tab);
        });
    });
    
    // Export conversation
    exportConversationBtn.addEventListener('click', exportConversation);
    
    // Clear conversation
    clearConversationBtn.addEventListener('click', clearConversation);
    
    // Expandable sections
    expandBtns.forEach(button => {
        button.addEventListener('click', function() {
            const content = this.nextElementSibling;
            content.classList.toggle('hidden');
            this.classList.toggle('expanded');
        });
    });
}

// Send a message
function sendMessage() {
    const query = chatInput.value.trim();
    if (!query) return;
    
    // Add message to UI
    addMessage(query, 'user');
    
    // Clear input
    chatInput.value = '';
    
    // Add to conversation history
    conversationHistory.push({
        role: 'user',
        content: query
    });
    
    // Update conversation count
    updateConversationCount();
    
    // Process query
    processQuery(query);
}

// Process query with API
async function processQuery(query) {
    try {
        // Add loading indicator
        const loadingEl = addLoadingIndicator();
        
        // Send request to API
        const response = await fetch(`${API_BASE_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query,
                session_id: sessionId
            })
        });
        
        const data = await response.json();
        
        // Remove loading indicator
        loadingEl.remove();
        
        if (data.error) {
            showToast(data.error, 'error');
            return;
        }
        
        // Add response to UI
        const answer = data.answer;
        addMessage(answer, 'assistant', data);
        
        // Add to conversation history
        conversationHistory.push({
            role: 'assistant',
            content: answer
        });
        
        // Update conversation count
        updateConversationCount();
        
    } catch (error) {
        console.error('Error processing query:', error);
        showToast('Error connecting to API server. Please try again.', 'error');
    }
}

// Add message to UI
function addMessage(text, role, data = null) {
    const messageEl = document.createElement('div');
    messageEl.className = `message ${role}`;
    messageEl.textContent = text;
    
    // Add to chat messages
    chatMessages.appendChild(messageEl);
    
    // Add metadata for assistant messages
    if (role === 'assistant' && data) {
        // Add confidence indicator, sources, etc.
        const metaEl = document.createElement('div');
        metaEl.className = 'response-meta';
        
        // Confidence
        if (data.confidence) {
            const confidenceEl = document.createElement('div');
            confidenceEl.className = 'response-section';
            confidenceEl.innerHTML = `
                <h4>Confidence</h4>
                <span class="confidence-indicator confidence-${data.confidence}">
                    ${capitalizeFirstLetter(data.confidence)}
                </span>
            `;
            metaEl.appendChild(confidenceEl);
        }
        
        // Sources
        if (data.sources && data.sources.length) {
            const sourcesEl = document.createElement('div');
            sourcesEl.className = 'response-section';
            sourcesEl.innerHTML = '<h4>Sources</h4>';
            
            const sourcesList = document.createElement('div');
            sourcesList.className = 'sources-list';
            
            data.sources.forEach(source => {
                const sourceEl = document.createElement('div');
                sourceEl.className = 'source-item';
                
                let sourceText = source.document;
                if (source.title) {
                    sourceText = source.title;
                } else if (source.original_filename) {
                    sourceText = source.original_filename;
                }
                
                if (source.page) {
                    sourceText += ` (Page ${source.page})`;
                }
                
                sourceEl.textContent = sourceText;
                sourcesList.appendChild(sourceEl);
            });
            
            sourcesEl.appendChild(sourcesList);
            metaEl.appendChild(sourcesEl);
        }
        
        // Metrics (optional)
        if (data.evaluation) {
            const metricsEl = document.createElement('div');
            metricsEl.className = 'response-section';
            metricsEl.innerHTML = '<h4>Evaluation Metrics</h4>';
            
            const metricsContainer = document.createElement('div');
            metricsContainer.className = 'metrics-container';
            
            Object.keys(data.evaluation).forEach(key => {
                if (key === 'total_score' || key === 'response_time') return;
                
                const value = data.evaluation[key];
                const metricEl = document.createElement('div');
                metricEl.className = 'metric-detail';
                
                const label = key.split('_').map(capitalizeFirstLetter).join(' ');
                
                metricEl.innerHTML = `
                    <span class="metric-detail-label">${label}</span>
                    <span class="metric-detail-value">${(value * 100).toFixed(0)}%</span>
                    <div class="progress-bar">
                        <div class="progress-bar-fill" style="width: ${value * 100}%"></div>
                    </div>
                `;
                
                metricsContainer.appendChild(metricEl);
            });
            
            metricsEl.appendChild(metricsContainer);
            metaEl.appendChild(metricsEl);
        }
        
        // Add feedback section
        const feedbackEl = document.createElement('div');
        feedbackEl.className = 'feedback-section';
        feedbackEl.innerHTML = `
            <h4>Feedback</h4>
            <div class="star-rating">
                <button class="star-btn" data-rating="1">★</button>
                <button class="star-btn" data-rating="2">★</button>
                <button class="star-btn" data-rating="3">★</button>
                <button class="star-btn" data-rating="4">★</button>
                <button class="star-btn" data-rating="5">★</button>
            </div>
            <textarea class="feedback-textarea" placeholder="Any corrections or suggestions?"></textarea>
            <button class="primary-btn submit-feedback-btn">Submit Feedback</button>
        `;
        
        // Add event listeners for star rating and feedback submission
        const starBtns = feedbackEl.querySelectorAll('.star-btn');
        const feedbackTextarea = feedbackEl.querySelector('.feedback-textarea');
        const submitBtn = feedbackEl.querySelector('.submit-feedback-btn');
        
        let selectedRating = 0;
        
        starBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const rating = parseInt(btn.dataset.rating);
                selectedRating = rating;
                
                starBtns.forEach(b => {
                    b.classList.toggle('selected', parseInt(b.dataset.rating) <= rating);
                });
            });
        });
        
        submitBtn.addEventListener('click', () => {
            if (selectedRating === 0) {
                showToast('Please select a rating first.', 'warning');
                return;
            }
            
            submitFeedback(text, data.answer, selectedRating, feedbackTextarea.value);
            feedbackEl.innerHTML = '<p class="success-message">Thank you for your feedback!</p>';
        });
        
        metaEl.appendChild(feedbackEl);
        
        // Append meta element to message
        messageEl.appendChild(metaEl);
    }
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Submit feedback to API
async function submitFeedback(query, answer, rating, feedbackText) {
    try {
        const response = await fetch(`${API_BASE_URL}/feedback`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query,
                answer,
                rating,
                feedback_text: feedbackText,
                session_id: sessionId
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            showToast(data.error, 'error');
            return;
        }
        
        showToast('Feedback submitted successfully!', 'success');
        
    } catch (error) {
        console.error('Error submitting feedback:', error);
        showToast('Error submitting feedback. Please try again.', 'error');
    }
}

// Handle file upload
function handleFileUpload() {
    const selectedFilesContainer = document.getElementById('selected-files-container');
    selectedFilesContainer.innerHTML = '';
    
    // Add newly selected files to our tracking array
    if (fileUpload.files.length > 0) {
        // Add new files to our selection array
        Array.from(fileUpload.files).forEach(file => {
            // Check if this file is already in our selection to avoid duplicates
            const isDuplicate = selectedFiles.some(f => 
                f.name === file.name && f.size === file.size && f.lastModified === file.lastModified
            );
            
            if (isDuplicate) {
                return; // Skip duplicate files
            }
            
            // Validate file before adding
            const validation = validatePdfFile(file);
            if (!validation.isValid) {
                showToast(`${file.name}: ${validation.message}`, 'warning');
                return;
            }
            
            // Add valid file to our selection
            selectedFiles.push(file);
        });
        
        // Reset the file input to prevent duplicate submissions
        fileUpload.value = '';
    }
    
    if (selectedFiles.length > 0) {
        processDocsBtn.disabled = false;
        
        // Display selected files
        selectedFiles.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'selected-file-item';
            
            // Validate file to show proper status
            const validation = validatePdfFile(file);
            const validationClass = validation.isValid ? 'file-valid' : 'file-invalid';
            const validationStatus = validation.isValid ? 'Ready to process' : validation.message;
            const fileIcon = validation.isValid ? '📄' : '⚠️';
            
            fileItem.innerHTML = `
                <div class="selected-file-header">
                    <span>${fileIcon} ${file.name}</span>
                    <button class="remove-file-btn" data-index="${index}" title="Remove file">×</button>
                </div>
                <div class="selected-file-details">
                    <div>Size: ${(file.size / 1024).toFixed(1)} KB</div>
                    <div class="${validationClass}">Status: ${validationStatus}</div>
                </div>
            `;
            
            selectedFilesContainer.appendChild(fileItem);
        });
        
        // Add event listeners to remove buttons
        const removeButtons = selectedFilesContainer.querySelectorAll('.remove-file-btn');
        removeButtons.forEach(btn => {
            btn.addEventListener('click', function(e) {
                e.preventDefault();
                removeSelectedFile(parseInt(this.dataset.index));
            });
        });
        
        // Show summary if multiple files
        if (selectedFiles.length > 1) {
            const summaryEl = document.createElement('div');
            summaryEl.className = 'selected-files-summary';
            
            summaryEl.innerHTML = `
                <span>${selectedFiles.length} files selected</span>
                <button id="clear-all-files" class="text-btn">Clear all</button>
            `;
            
            selectedFilesContainer.appendChild(summaryEl);
            
            // Add event listener for clear all button
            document.getElementById('clear-all-files').addEventListener('click', clearAllSelectedFiles);
        }
    } else {
        processDocsBtn.disabled = true;
    }
}

// Remove a file from the selection
function removeSelectedFile(index) {
    // Remove file at specific index from our array
    if (index >= 0 && index < selectedFiles.length) {
        selectedFiles.splice(index, 1);
    }
    
    // Update the UI
    handleFileUpload();
}

// Clear all selected files
function clearAllSelectedFiles() {
    // Clear our array
    selectedFiles = [];
    
    // Reset the file input for good measure
    fileUpload.value = '';
    
    // Update the UI
    handleFileUpload();
}

// Validate a PDF file before upload
function validatePdfFile(file) {
    // Check if it's a PDF file
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        return {
            isValid: false,
            message: "Only PDF files are allowed"
        };
    }
    
    // Check if file is empty
    if (file.size === 0) {
        return {
            isValid: false,
            message: "File is empty"
        };
    }
    
    // Check file size (limit to 50MB)
    const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
    if (file.size > MAX_FILE_SIZE) {
        return {
            isValid: false,
            message: "File size must be less than 50MB"
        };
    }
    
    return {
        isValid: true,
        message: "File is valid"
    };
}

// Process documents
async function processDocuments() {
    if (!selectedFiles || selectedFiles.length === 0) {
        showToast('No files selected', 'warning');
        return;
    }
    
    // Check if all files are valid
    let hasInvalidFile = false;
    for (let i = 0; i < selectedFiles.length; i++) {
        const validation = validatePdfFile(selectedFiles[i]);
        if (!validation.isValid) {
            hasInvalidFile = true;
            showToast(`Error: ${selectedFiles[i].name} - ${validation.message}`, 'error');
        }
        
        // Double-check file content by reading a small sample
        try {
            const fileSlice = selectedFiles[i].slice(0, 1024); // Read first 1KB to verify content
            const reader = new FileReader();
            
            // Use promise to make it work with async/await
            const hasContent = await new Promise((resolve) => {
                reader.onloadend = function(e) {
                    // Check if we got any actual content
                    resolve(e.target.result && e.target.result.byteLength > 0);
                };
                reader.readAsArrayBuffer(fileSlice);
            });
            
            if (!hasContent) {
                hasInvalidFile = true;
                showToast(`Error: ${selectedFiles[i].name} - File appears to be empty or corrupted`, 'error');
            }
        } catch (error) {
            console.error(`Error checking file content for ${selectedFiles[i].name}:`, error);
        }
    }
    
    if (hasInvalidFile) {
        return;
    }
    
    // Disable button and show processing state
    const originalText = processDocsBtn.textContent;
    processDocsBtn.disabled = true;
    processDocsBtn.textContent = 'Processing...';
    
    // Save the selected files container content for reference
    const selectedFilesContainer = document.getElementById('selected-files-container');
    const selectedFilesHtml = selectedFilesContainer.innerHTML;
    
    try {
        // Process each file
        for (let i = 0; i < selectedFiles.length; i++) {
            const file = selectedFiles[i];
            
            // Update button text to show current file
            processDocsBtn.textContent = `Processing ${i+1}/${selectedFiles.length}...`;
            
            // Show progress toast and update UI with percentage
            const progressText = `Processing ${file.name} (${i+1}/${selectedFiles.length})...`;
            const progressPercent = Math.round(((i + 0.5) / selectedFiles.length) * 100);
            
            showToast(`📄 ${progressText} ${progressPercent}%`, 'info');
            
            // Add a temporary processing file indicator to the UI using our helper function
            updateDocumentStatus(file, 'processing');
            
            // Create new FormData for each file to ensure clean state
            const formData = new FormData();
            
            // Create a new Blob from the File to ensure content is preserved
            const fileBlob = new Blob([file], { type: 'application/pdf' });
            
            // Add to FormData with original filename
            formData.append('file', fileBlob, file.name);
            
            console.log('Uploading file:', file.name, 'Size:', file.size, 'Type:', file.type);
            
            // Upload and process file with correct content type
            const response = await fetch(`${API_BASE_URL}/upload`, {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.error) {
                console.error('File upload error:', data.error);
                showToast(`Error: ${data.error}`, 'error');
                
                // Update UI to show error
                updateDocumentStatus(file, 'error', `Error: ${data.error}`);
                continue;
            }
            
            // Add to processed files
            if (data.file_info) {
                console.log('File processed successfully:', data.file_info);
                
                // Update status to success
                updateDocumentStatus(file, 'success', 'Successfully processed');
                
                // Update with real data from API
                await updateProcessedFilesList();
                showToast(`✅ ${file.name} processed successfully!`, 'success');
            } else {
                console.warn('No file_info in response:', data);
                
                // Update UI to show warning
                updateDocumentStatus(file, 'error', 'No file data returned from server');
            }
        }
        
        // Update system status
        await checkSystemStatus();
        
        // Reset file input, button, and selected files container
        const fileCount = selectedFiles.length;
        fileUpload.value = '';
        selectedFiles = []; // Clear our selected files array
        document.getElementById('selected-files-container').innerHTML = '';
        processDocsBtn.disabled = true;
        processDocsBtn.textContent = originalText;
        
        showToast(`🎉 All ${fileCount} documents processed successfully!`, 'success');
        
    } catch (error) {
        console.error('Error processing documents:', error);
        showToast('Error connecting to API server. Please try again.', 'error');
    } finally {
        // Restore button state
        processDocsBtn.textContent = originalText;
        processDocsBtn.disabled = fileUpload.files.length === 0;
    }
}

// Update document processing status
function updateDocumentStatus(file, status = 'processing', message = null) {
    const statusClass = {
        'processing': 'file-processing',
        'success': 'file-success',
        'error': 'error-message'
    };
    
    // Look for existing temporary element for this file
    const existingElements = Array.from(processedFilesList.children);
    const tempElement = existingElements.find(el => 
        el.textContent.includes(file.name) && 
        el.classList.contains('file-processing')
    );
    
    if (tempElement) {
        // Update existing element
        tempElement.className = `file-item ${statusClass[status] || 'file-processing'}`;
        
        const headerSpan = tempElement.querySelector('.file-item-header span');
        const detailsDiv = tempElement.querySelector('.file-item-details');
        
        if (headerSpan) {
            const icon = status === 'processing' ? '⏳' : status === 'success' ? '✅' : '❌';
            headerSpan.innerHTML = `${icon} ${file.name}`;
        }
        
        if (detailsDiv) {
            if (message) {
                detailsDiv.innerHTML = `<div>${message}</div>
                    <div>Size: ${(file.size / 1024).toFixed(1)} KB</div>`;
            } else if (status === 'success') {
                detailsDiv.innerHTML = `<div>Successfully processed</div>
                    <div>Size: ${(file.size / 1024).toFixed(1)} KB</div>`;
            }
        }
    } else {
        // Create a new status element
        const newElement = document.createElement('div');
        newElement.className = `file-item ${statusClass[status] || 'file-processing'}`;
        
        const icon = status === 'processing' ? '⏳' : status === 'success' ? '✅' : '❌';
        const statusMessage = message || (status === 'processing' ? 'Processing... please wait' : 
                              status === 'success' ? 'Successfully processed' : 'Error processing file');
        
        newElement.innerHTML = `
            <div class="file-item-header">
                <span>${icon} ${file.name}</span>
            </div>
            <div class="file-item-details">
                <div>${statusMessage}</div>
                <div>Size: ${(file.size / 1024).toFixed(1)} KB</div>
            </div>
        `;
        
        processedFilesList.appendChild(newElement);
    }
}

// Update processed files list
async function updateProcessedFilesList() {
    try {
        console.log('Updating processed files list...');
        // Show loading state
        processedFilesList.innerHTML = '<div class="loading">Loading files...</div>';
        
        // Fetch processed files from API
        const response = await fetch(`${API_BASE_URL}/files`);
        if (!response.ok) {
            throw new Error(`API responded with status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            console.error('Error fetching processed files:', data.error);
            processedFilesList.innerHTML = '<div class="error-message">Error loading processed files</div>';
            return;
        }
        
        // Update global variable
        processedFiles = data.files || [];
        
        // Clear the list
        processedFilesList.innerHTML = '';
        
        // Display empty state if no files
        if (processedFiles.length === 0) {
            processedFilesList.innerHTML = '<div class="empty-state">No documents processed yet</div>';
            return;
        }
        
        // Display each file
        processedFiles.forEach((file, index) => {
            const fileEl = document.createElement('div');
            fileEl.className = 'file-item file-success';
            
            const icon = file.auto_loaded ? '🔄' : '📄';
            
            fileEl.innerHTML = `
                <div class="file-item-header">
                    <span>${icon} ${file.filename}</span>
                </div>
                <div class="file-item-details">
                    <div><strong>Size:</strong> ${(file.size / 1024).toFixed(1)} KB</div>
                    <div><strong>Chunks:</strong> ${file.chunks}</div>
                    <div><strong>Extraction:</strong> ${file.extraction_method}</div>
                    <div><strong>Processed:</strong> ${file.timestamp}</div>
                    ${file.auto_loaded ? '<div><strong>Source:</strong> Auto-loaded from documents directory</div>' : ''}
                </div>
            `;
            
            processedFilesList.appendChild(fileEl);
        });
    } catch (error) {
        console.error('Error updating processed files list:', error);
        processedFilesList.innerHTML = '<div class="error-message">Error loading processed files: ' + error.message + '</div>';
    }
    
    // Update document count in UI
    if (processedFiles) {
        docCount.textContent = processedFiles.length;
        analyticsDocCount.textContent = processedFiles.length;
    }
}

// Clear knowledge base
async function clearKnowledgeBase() {
    try {
        const response = await fetch(`${API_BASE_URL}/collection`, {
            method: 'DELETE'
        });
        
        const data = await response.json();
        
        if (data.error) {
            showToast(data.error, 'error');
            return;
        }
        
        // Reset state
        processedFiles = [];
        await updateProcessedFilesList();
        
        // Reset confirmation UI
        confirmClearCheckbox.checked = false;
        confirmClear.classList.add('hidden');
        confirmClearBtn.disabled = true;
        
        // Update system status
        checkSystemStatus();
        
        showToast('Knowledge base cleared successfully!', 'success');
        
    } catch (error) {
        console.error('Error clearing knowledge base:', error);
        showToast('Error connecting to API server. Please try again.', 'error');
    }
}

// Show a modal
function showModal(modal) {
    modalOverlay.classList.remove('hidden');
    modal.classList.remove('hidden');
}

// Close all modals
function closeModals() {
    modalOverlay.classList.add('hidden');
    document.querySelectorAll('.modal').forEach(modal => {
        modal.classList.add('hidden');
    });
}

// Update conversation count
function updateConversationCount() {
    const count = Math.floor(conversationHistory.length / 2);
    conversationCountEl.textContent = count;
}

// Export conversation
function exportConversation() {
    if (conversationHistory.length === 0) {
        showToast('No conversation to export.', 'warning');
        return;
    }
    
    const json = JSON.stringify(conversationHistory, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `conversation_${sessionId}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// Clear conversation
function clearConversation() {
    conversationHistory = [];
    chatMessages.innerHTML = '';
    updateConversationCount();
    showToast('Conversation history cleared!', 'success');
}

// Switch tab
function switchTab(tabId) {
    // Deactivate all tabs
    tabBtns.forEach(btn => {
        btn.classList.remove('active');
    });
    
    tabPanes.forEach(pane => {
        pane.classList.remove('active');
    });
    
    // Activate selected tab
    document.querySelector(`.tab-btn[data-tab="${tabId}"]`).classList.add('active');
    document.getElementById(`${tabId}-tab`).classList.add('active');
}

// Add loading indicator
function addLoadingIndicator() {
    const loaderEl = document.createElement('div');
    loaderEl.className = 'loader-container';
    loaderEl.innerHTML = '<div class="loader"></div>';
    chatMessages.appendChild(loaderEl);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return loaderEl;
}

// Show a toast message
function showToast(message, type = 'info') {
    // Create toast container if it doesn't exist
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container';
        document.body.appendChild(toastContainer);
    }
    
    // Create toast
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    
    // Add toast to container
    toastContainer.appendChild(toast);
    
    // Remove toast after 3 seconds
    setTimeout(() => {
        toast.classList.add('toast-fade-out');
        setTimeout(() => {
            toast.remove();
        }, 300);
    }, 3000);
}

// Helper function to capitalize first letter
function capitalizeFirstLetter(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
}

// Add toast styles
const style = document.createElement('style');
style.textContent = `
    .toast-container {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    
    .toast {
        padding: 10px 20px;
        border-radius: 4px;
        color: white;
        opacity: 1;
        transition: opacity 0.3s ease;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    
    .toast-info {
        background-color: #17a2b8;
    }
    
    .toast-success {
        background-color: #28a745;
    }
    
    .toast-warning {
        background-color: #ffc107;
        color: #333;
    }
    
    .toast-error {
        background-color: #dc3545;
    }
    
    .toast-fade-out {
        opacity: 0;
    }
`;
document.head.appendChild(style);

// Initialize on DOM content loaded
document.addEventListener('DOMContentLoaded', init);
