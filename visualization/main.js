const statusElement = document.getElementById('status');
const taskDescriptionElement = document.getElementById('task_description'); // Assuming you might add this later
const btnNextLLMTask = document.getElementById('btnNextLLMTask');

let socket;

function connectWebSocket() {
    // ... existing connectWebSocket function ...
    socket.onopen = () => {
        console.log("WebSocket connected.");
        statusElement.textContent = "Connected to Physics Server!";
        statusElement.style.color = "green";
        if (btnNextLLMTask) btnNextLLMTask.disabled = false; // Enable button on connect
    };

    socket.onmessage = (event) => {
        // ... existing onmessage handler ...
    };

    socket.onerror = (error) => {
        console.error("WebSocket Error: ", error);
        statusElement.textContent = "Error connecting to server.";
        statusElement.style.color = "red";
        if (btnNextLLMTask) btnNextLLMTask.disabled = true; // Disable button on error
        // Optional: attempt to reconnect here or in onclose
    };

    socket.onclose = () => {
        console.log("WebSocket disconnected. Reconnecting in 3 seconds...");
        statusElement.textContent = "Disconnected. Attempting to reconnect in 3s...";
        statusElement.style.color = "orange";
        if (btnNextLLMTask) btnNextLLMTask.disabled = true; // Disable button on close
        setTimeout(connectWebSocket, 3000);
    };
}

// Add event listener for the new button
if (btnNextLLMTask) {
    btnNextLLMTask.disabled = true; // Initially disabled until WebSocket connects
    btnNextLLMTask.addEventListener('click', () => {
        if (socket && socket.readyState === WebSocket.OPEN) {
            console.log("Requesting next LLM-driven task...");
            socket.send(JSON.stringify({ command: "next_llm_task" }));
            // Optionally disable the button temporarily to prevent spamming
            btnNextLLMTask.disabled = true;
            // Re-enable after a delay, or based on a response from the server if implemented
            setTimeout(() => {
                if (socket && socket.readyState === WebSocket.OPEN) {
                    btnNextLLMTask.disabled = false;
                }
            }, 3000); // Re-enable after 3s, adjust as needed
        } else {
            console.warn("WebSocket not open. Cannot send command.");
        }
    });
}

// Initial connection attempt
connectWebSocket();

// ... rest of your main.js (init, animate, etc.)
