document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('query-form');
    const chatBox = document.getElementById('chat-box');
    const sessionIdInput = document.getElementById('session_id');
    const progressPanel = document.getElementById('progress-panel');
    const mainContent = document.querySelector('.main-content');
    const estimatedTime = document.getElementById('estimated-time');
    const toggleBtn = document.getElementById('toggle-progress-btn');
    const currentStage = document.getElementById('current-stage');
    const closeBtn = document.getElementById('close-btn');

    const stageElements = {
        planning: document.querySelector('[data-stage="planning"]'),
        sql: document.querySelector('[data-stage="sql"]'),
        query: document.querySelector('[data-stage="query"]'),
        analysis: document.querySelector('[data-stage="analysis"]'),
        final: document.querySelector('[data-stage="final"]')
    };

    const statusElements = {
        planning: document.getElementById('status-planning'),
        sql: document.getElementById('status-sql'),
        query: document.getElementById('status-query'),
        analysis: document.getElementById('status-analysis'),
        final: document.getElementById('status-final')
    };

    let isProcessing = false;

    if (!sessionIdInput.value) {
        sessionIdInput.value = generateUUID();
    }

    if (typeof marked !== 'undefined') {
        marked.setOptions({
            breaks: true,
            gfm: true,
            headerIds: false,
            mangle: false
        });
    }

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        if (isProcessing) return;
        
        const formData = new FormData(form);
        const query = formData.get('query');
        
        appendMessage(query, 'user');
        form.querySelector('input[name="query"]').value = '';
        
        showProgressPanel();

        try {
            const response = await fetch('/send_query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded'
                },
                body: new URLSearchParams({
                    query: query,
                    session_id: sessionIdInput.value
                })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();

                for (const line of lines) {
                    if (line.trim() === '') continue;
                    try {
                        const data = JSON.parse(line);
                        handleStreamedData(data);
                    } catch (e) {
                        console.error('Error parsing NDJSON line:', e, 'Line:', line);
                    }
                }
            }
            
            completeProcessing();
        } catch (error) {
            console.error('Error during processing:', error);
            handleProcessingError(error);
        }
    });

    closeBtn.addEventListener('click', () => {
        hideProgressPanel();
    });

    toggleBtn.addEventListener('click', () => {
        progressPanel.classList.toggle('show');
    });

    function handleStreamedData(data) {
        if (data.stage) {
            updateProgress(data.stage);
        }

        if (data.response) {
            updateAssistantMessage(data.response, true);
        } else if (data.content) {
            updateAssistantMessage(data.content);
        }
    }

    function updateProgress(stage) {
        const stageMap = {
            "planning": { element: stageElements.planning, status: statusElements.planning, description: "Planning Analysis..." },
            "sql": { element: stageElements.sql, status: statusElements.sql, description: "Generating SQL Queries..." },
            "query": { element: stageElements.query, status: statusElements.query, description: "Executing Database Query..." },
            "analysis": { element: stageElements.analysis, status: statusElements.analysis, description: "Analyzing Financial Data..." },
            "final": { element: stageElements.final, status: statusElements.final, description: "Creating Investment Report..." }
        };

        Object.keys(stageMap).forEach(key => {
            stageMap[key].element.classList.remove('active', 'completed');
            stageMap[key].status.textContent = '⏳';
        });

        if (stageMap[stage]) {
            stageMap[stage].element.classList.add('active');
            stageMap[stage].status.textContent = '🔄';
            updateCurrentStage(stageMap[stage].description);

            // Mark previous stages as completed
            const stages = Object.keys(stageMap);
            const currentIndex = stages.indexOf(stage);
            for (let i = 0; i < currentIndex; i++) {
                const prevStageKey = stages[i];
                stageMap[prevStageKey].element.classList.remove('active');
                stageMap[prevStageKey].element.classList.add('completed');
                stageMap[prevStageKey].status.textContent = '✅';
            }
        }
    }

    function appendMessage(content, role) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', `${role}-message`);
        
        if (role === 'user') {
            messageElement.innerText = content;
        } else {
            try {
                if (typeof marked !== 'undefined' && content) {
                    messageElement.innerHTML = marked.parse(content);
                } else {
                    messageElement.innerText = content;
                }
            } catch (error) {
                console.warn('Markdown parsing failed:', error);
                messageElement.innerText = content;
            }
        }
        
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    let assistantMessageElement = null;
    function updateAssistantMessage(content, isFinal = false) {
        if (!assistantMessageElement) {
            assistantMessageElement = document.createElement('div');
            assistantMessageElement.classList.add('message', 'assistant-message');
            chatBox.appendChild(assistantMessageElement);
        }
        
        try {
            if (typeof marked !== 'undefined' && content) {
                assistantMessageElement.innerHTML = marked.parse(content);
            } else {
                assistantMessageElement.innerText = content;
            }
        } catch (error) {
            console.warn('Markdown parsing failed during update:', error);
            assistantMessageElement.innerText = content;
        }
        
        chatBox.scrollTop = chatBox.scrollHeight;

        if (isFinal) {
            assistantMessageElement = null;
            hideProgressPanel();
        }
    }

    function generateUUID() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }

    function showProgressPanel() {
        progressPanel.style.display = 'flex';
        setTimeout(() => {
            progressPanel.classList.add('show');
        }, 10);
        isProcessing = true;
        updateProgress('planning');
        estimatedTime.textContent = 'Processing in progress...';
    }

    function hideProgressPanel() {
        progressPanel.classList.remove('show');
        isProcessing = false;
    }

    function updateCurrentStage(stageName) {
        currentStage.textContent = stageName;
    }

    function completeProcessing() {
        updateProgress('final');
        stageElements.final.classList.add('completed');
        statusElements.final.textContent = '✅';
        updateCurrentStage('Complete!');
        estimatedTime.textContent = 'Analysis finished!';
        hideProgressPanel();
        isProcessing = false;
    }

    function cancelProcessing() {
        isProcessing = false;
        updateCurrentStage('Cancelled');
        estimatedTime.textContent = 'Processing cancelled';
        setTimeout(hideProgressPanel, 1000);
    }

    function handleProcessingError(error) {
        updateCurrentStage('Error occurred');
        estimatedTime.textContent = 'Processing failed';
        console.error('Processing error:', error);
        appendMessage('Sorry, there was an error processing your request. Please try again.', 'assistant');
        setTimeout(hideProgressPanel, 2000);
    }
});
