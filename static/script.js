/**
 * This file manages the drawing canvas, drawing history (undo/redo), and user interface interactions.
 * Layer management has been removed for simplification.
 */
class CanvasManager {
    /**
     * Initializes the CanvasManager.
     * @param {string} canvasId - The ID of the HTML canvas element.
     */
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId); // Gets the canvas element from the DOM.
        this.ctx = this.canvas.getContext("2d"); // Gets the 2D rendering context for drawing.
        this.pathHistory = []; // Stores canvas states for undo functionality.
        this.redoHistory = []; // Stores undone states for redo functionality.
        this.listeners = {}; // Custom event listeners for inter-component communication.
        this.resizing = false; // Flag to prevent rapid resizing issues.

        // Sets initial canvas size to window dimensions.
        this.resizeCanvas();
        // Adds event listener for window resizing.
        window.addEventListener("resize", () => {
            this.resizing = true;
            // Debounces the resize operation for performance.
            setTimeout(() => {
                this.resizeCanvas();
                this.resizing = false;
            }, 150);
        });
        this.saveState(); // Saves the initial blank canvas state.
    }

    /**
     * Gets the 2D rendering context of the canvas.
     * @returns {CanvasRenderingContext2D} The context of the main canvas.
     */
    getActiveContext() {
        return this.ctx; // Always returns the main context as layers are removed.
    }

    /**
     * Resizes the main canvas to fit the window dimensions and redraws existing content.
     */
    resizeCanvas() {
        // Creates a temporary canvas to hold current drawing before resizing.
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = this.canvas.width;
        tempCanvas.height = this.canvas.height;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(this.canvas, 0, 0); // Copies current canvas content.

        // Updates main canvas dimensions.
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        // Draws the saved content back onto the resized main canvas.
        this.ctx.drawImage(tempCanvas, 0, 0);
        this.trigger("canvasresized"); // Triggers a custom event notifying other components that the canvas has been resized.
        this.saveState(); // Saves state after resize if content was present.
    }

    /**
     * Saves the current state of the main canvas to the history stack for undo functionality.
     * This should be called after a completed drawing action.
     */
    saveState() {
        // Stores a snapshot of the current main canvas image data.
        this.pathHistory.push(this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height));
        this.redoHistory = []; // Clears redo history when a new state is saved.
        // Limits history size to prevent excessive memory usage.
        if (this.pathHistory.length > 50) { // Keeps the history limit reasonable.
            this.pathHistory.shift(); // Removes the oldest state.
        }
        this.trigger("historyupdated"); // Notifies the UI about history changes.
    }

    /**
     * Restores the canvas to a specific image data state.
     * @param {ImageData} imageData - The ImageData object to restore.
     */
    restoreState(imageData) {
        this.ctx.putImageData(imageData, 0, 0); // Puts image data onto the main canvas.
        this.trigger("historyupdated"); // Notifies the UI about history changes.
    }

    /**
     * Undoes the last drawing action by restoring the previous state from history.
     */
    undo() {
        if (this.pathHistory.length > 1) { // Ensures there's at least one previous state to revert to.
            this.redoHistory.push(this.pathHistory.pop()); // Moves the current state to redo history.
            this.restoreState(this.pathHistory[this.pathHistory.length - 1]); // Restores the previous state.
        }
    }

    /**
     * Redoes the last undone action by restoring the next state from redo history.
     */
    redo() {
        if (this.redoHistory.length > 0) { // Ensures there are states to redo.
            this.pathHistory.push(this.redoHistory.pop()); // Moves the state from redo to main history.
            this.restoreState(this.pathHistory[this.pathHistory.length - 1]); // Restores that state.
        }
    }

    /**
     * Clears the entire canvas and resets drawing history.
     */
    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height); // Clears all pixels on the main canvas.
        this.pathHistory = []; // Resets history.
        this.redoHistory = []; // Resets redo history.
        this.saveState(); // Saves the cleared state.
    }

    /**
     * Registers a custom event listener.
     * @param {string} eventName - The name of the event to listen for.
     * @param {function} listener - The callback function to execute when the event is triggered.
     */
    on(eventName, listener) {
        this.listeners[eventName] = this.listeners[eventName] || [];
        this.listeners[eventName].push(listener);
    }

    /**
     * Triggers a custom event, notifying all registered listeners.
     * @param {string} eventName - The name of the event to trigger.
     * @param {...any} args - Arguments to pass to the listeners.
     */
    trigger(eventName, ...args) {
        if (this.listeners[eventName]) {
            this.listeners[eventName].forEach(listener => listener.apply(this, args));
        }
    }

    /**
     * Gets the current canvas content as a Data URL.
     * @param {string} format - The image format ('png', 'jpeg').
     * @returns {string} The Data URL of the canvas image.
     */
    getImageData(format = 'png') {
        return this.canvas.toDataURL(`image/${format}`);
    }
}

/**
 * Base class for all drawing tools.
 */
class Tool {
    /**
     * Initializes a Tool.
     * @param {CanvasManager} canvasManager - The canvas manager instance.
     * @param {string} name - The name of the tool.
     */
    constructor(canvasManager, name) {
        this.canvasManager = canvasManager;
        this.name = name;
        this.isActive = false; // Flag to indicate if the tool is currently active.
    }

    /**
     * Activates the tool, setting it as the current active tool and notifying the manager.
     */
    activate() {
        this.isActive = true;
        this.canvasManager.trigger("toolchanged", this.name); // Notifies the UI that the tool has changed.
    }

    /**
     * Deactivates the tool.
     */
    deactivate() {
        this.isActive = false;
    }

    /**
     * Handles various mouse/touch events. To be implemented by subclasses.
     * @param {Event} event - The DOM event object.
     */
    handleEvent(event) {
        // This method should be implemented by specific tool subclasses.
    }
}

/**
 * Pen tool for freehand drawing.
 */
class PenTool extends Tool {
    /**
     * @param {CanvasManager} canvasManager - The canvas manager instance.
     */
    constructor(canvasManager) {
        super(canvasManager, "pen");
        this.isDrawing = false; // Flag to track if drawing is in progress.
        this.lastX = 0; // Last X coordinate.
        this.lastY = 0; // Last Y coordinate.
        this.color = "#333"; // Default pen color.
        this.size = 5; // Default pen size.
    }

    /**
     * Activates the pen tool, setting context properties.
     */
    activate() {
        super.activate();
        const ctx = this.canvasManager.getActiveContext();
        ctx.strokeStyle = this.color; // Sets the stroke color.
        ctx.lineWidth = this.size; // Sets the line width.
        ctx.lineCap = "round"; // Sets rounded line caps.
        ctx.lineJoin = "round"; // Sets rounded line joins.
    }

    /**
     * Sets the pen color.
     * @param {string} color - The new color (e.g., "#RRGGBB").
     */
    setColor(color) {
        this.color = color;
        // Applies only if the tool is active or being initialized.
        if (this.isActive || !this.canvasManager.getActiveContext().strokeStyle) { 
            const ctx = this.canvasManager.getActiveContext();
            if (ctx) { 
                ctx.strokeStyle = this.color;
            }
        }
    }

    /**
     * Sets the pen size.
     * @param {number} size - The new line width.
     */
    setSize(size) {
        this.size = size;
        // Applies only if the tool is active or being initialized.
        if (this.isActive || !this.canvasManager.getActiveContext().lineWidth) {
            const ctx = this.canvasManager.getActiveContext();
            if (ctx) {
                ctx.lineWidth = this.size;
            }
        }
    }

    /**
     * Handles mouse events for drawing with the pen.
     * @param {MouseEvent} event - The mouse event.
     */
    handleEvent(event) {
        const ctx = this.canvasManager.getActiveContext();
        const x = event.clientX;
        const y = event.clientY;

        if (event.type === "mousedown") {
            this.isDrawing = true; // Drawing started.
            [this.lastX, this.lastY] = [x, y]; // Sets last coordinates.
            ctx.beginPath(); // Starts a new path.
            ctx.moveTo(this.lastX, this.lastY); // Moves to the starting point.
        } else if (event.type === "mousemove") {
            if (!this.isDrawing) return; // Only draws if mouse is down.
            ctx.lineTo(x, y); // Draws a line to the current position.
            ctx.stroke(); // Renders the line.
            [this.lastX, this.lastY] = [x, y]; // Updates last position.
        } else if (event.type === "mouseup" || event.type === "mouseout") {
            if (this.isDrawing) {
                this.isDrawing = false; // Drawing finished.
                this.canvasManager.saveState(); // Saves state after drawing stops.
            }
        }
    }
}

/**
 * Eraser tool for removing pixels from the canvas.
 */
class EraserTool extends Tool {
    /**
     * @param {CanvasManager} canvasManager - The canvas manager instance.
     */
    constructor(canvasManager) {
        super(canvasManager, "eraser");
        this.isErasing = false; // Flag to track if erasing is in progress.
        this.size = 10; // Default eraser size.
        this.cursor = document.getElementById("eraserCursor"); // Reference to the custom cursor element.
    }

    /**
     * Activates the eraser tool, showing the custom cursor.
     */
    activate() {
        super.activate();
        this.cursor.style.display = "block"; // Shows the eraser cursor.
        // Ensures cursor size is applied on activation.
        this.cursor.style.width = `${this.size}px`;
        this.cursor.style.height = `${this.size}px`;
    }

    /**
     * Deactivates the eraser tool, hiding the custom cursor.
     */
    deactivate() {
        super.deactivate();
        this.cursor.style.display = "none"; // Hides the eraser cursor.
    }

    /**
     * Sets the eraser size and updates the custom cursor.
     * @param {number} size - The new eraser size.
     */
    setSize(size) {
        this.size = size;
        // Updates the visual size of the custom eraser cursor.
        this.cursor.style.width = `${this.size}px`;
        this.cursor.style.height = `${this.size}px`;
    }

    /**
     * Handles mouse events for erasing.
     * @param {MouseEvent} event - The mouse event.
     */
    handleEvent(event) {
        const ctx = this.canvasManager.getActiveContext();
        const x = event.clientX;
        const y = event.clientY;
        const eraserOffset = this.size / 2; // Offset to center the eraser cursor.

        // Positions the custom cursor.
        this.cursor.style.left = `${x - eraserOffset}px`;
        this.cursor.style.top = `${y - eraserOffset}px`;

        if (event.type === "mousedown") {
            this.isErasing = true; // Erasing started.
        } else if (event.type === "mousemove" && this.isErasing) { 
            // Clears a rectangular area on the canvas where the eraser moves.
            ctx.clearRect(x - eraserOffset, y - eraserOffset, this.size, this.size);
        } else if (event.type === "mouseup" || event.type === "mouseout") {
            if (this.isErasing) { 
                this.isErasing = false; // Erasing finished.
                this.canvasManager.saveState(); // Saves state after erasing stops.
            }
        }
    }
}

/**
 * Line tool for drawing straight lines.
 */
class LineTool extends Tool {
    /**
     * @param {CanvasManager} canvasManager - The canvas manager instance.
     */
    constructor(canvasManager) {
        super(canvasManager, "line");
        this.isDrawing = false;
        this.startX = 0; // Starting X coordinate.
        this.startY = 0; // Starting Y coordinate.
        this.color = "#333";
        this.size = 5;
        this.tempImageData = null; // Specific to LineTool for drawing preview.
    }

    /**
     * Activates the line tool, setting context properties.
     */
    activate() {
        super.activate();
        const ctx = this.canvasManager.getActiveContext();
        ctx.strokeStyle = this.color;
        ctx.lineWidth = this.size;
        ctx.lineCap = "round";
    }

    /**
     * Sets the line color.
     * @param {string} color - The new color.
     */
    setColor(color) {
        this.color = color;
        if (this.isActive) {
            this.canvasManager.getActiveContext().strokeStyle = this.color;
        }
    }

    /**
     * Sets the line size.
     * @param {number} size - The new line width.
     */
    setSize(size) {
        this.size = size;
        if (this.isActive) {
            this.canvasManager.getActiveContext().lineWidth = this.size;
        }
    }

    /**
     * Handles mouse events for drawing lines.
     * @param {MouseEvent} event - The mouse event.
     */
    handleEvent(event) {
        const ctx = this.canvasManager.getActiveContext();
        const x = event.clientX;
        const y = event.clientY;

        if (event.type === "mousedown") {
            this.isDrawing = true; // Drawing started.
            [this.startX, this.startY] = [x, y]; // Sets starting point.
            // Saves current canvas state to restore for temporary drawing preview.
            this.tempImageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
        } else if (event.type === "mousemove") {
            if (!this.isDrawing) return;
            // Restores the saved state to clear the previous temporary line before drawing a new one.
            ctx.putImageData(this.tempImageData, 0, 0);
            ctx.beginPath();
            ctx.moveTo(this.startX, this.startY); // Moves to the start point.
            ctx.lineTo(x, y); // Draws a line to the current position.
            ctx.stroke(); // Renders the temporary line.
        } else if (event.type === "mouseup" || event.type === "mouseout") {
            if (this.isDrawing) {
                this.isDrawing = false;
                this.tempImageData = null; // Clears temporary image data.
                this.canvasManager.saveState(); // Saves state after the final line is drawn.
            }
        }
    }
}

/**
 * Rectangle tool for drawing rectangles.
 */
class RectTool extends Tool {
    /**
     * @param {CanvasManager} canvasManager - The canvas manager instance.
     */
    constructor(canvasManager) {
        super(canvasManager, "rect");
        this.isDrawing = false;
        this.startX = 0;
        this.startY = 0;
        this.color = "#333";
        this.size = 5;
        this.tempImageData = null; // Specific to RectTool for drawing preview.
    }

    /**
     * Activates the rectangle tool, setting context properties.
     */
    activate() {
        super.activate();
        const ctx = this.canvasManager.getActiveContext();
        ctx.strokeStyle = this.color;
        ctx.lineWidth = this.size;
    }

    /**
     * Sets the rectangle stroke color.
     * @param {string} color - The new color.
     */
    setColor(color) {
        this.color = color;
        if (this.isActive) {
            this.canvasManager.getActiveContext().strokeStyle = this.color;
        }
    }

    /**
     * Sets the rectangle stroke size.
     * @param {number} size - The new line width.
     */
    setSize(size) {
        this.size = size;
        if (this.isActive) {
            this.canvasManager.getActiveContext().lineWidth = this.size;
        }
    }

    /**
     * Handles mouse events for drawing rectangles.
     * @param {MouseEvent} event - The mouse event.
     */
    handleEvent(event) {
        const ctx = this.canvasManager.getActiveContext();
        const x = event.clientX;
        const y = event.clientY;

        if (event.type === "mousedown") {
            this.isDrawing = true; // Drawing started.
            [this.startX, this.startY] = [x, y];
            // Saves current canvas state to restore for temporary drawing preview.
            this.tempImageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
        } else if (event.type === "mousemove") {
            if (!this.isDrawing) return;
            // Restores the saved state to clear the previous temporary rectangle before drawing a new one.
            ctx.putImageData(this.tempImageData, 0, 0);
            // Draws the rectangle from the start point to the current mouse position.
            ctx.strokeRect(this.startX, this.startY, x - this.startX, y - this.startY);
        } else if (event.type === "mouseup" || event.type === "mouseout") {
            if (this.isDrawing) {
                this.isDrawing = false;
                this.tempImageData = null; // Clears temporary image data.
                this.canvasManager.saveState(); // Saves state after the final rectangle is drawn.
            }
        }
    }
}

/**
 * Circle/Ellipse tool for drawing circular shapes.
 */
class CircleTool extends Tool {
    /**
     * @param {CanvasManager} canvasManager - The canvas manager instance.
     */
    constructor(canvasManager) {
        super(canvasManager, "circle");
        this.isDrawing = false;
        this.startX = 0;
        this.startY = 0;
        this.color = "#333";
        this.size = 5;
        this.tempImageData = null; // Specific to CircleTool for drawing preview.
    }

    /**
     * Activates the circle tool, setting context properties.
     */
    activate() {
        super.activate();
        const ctx = this.canvasManager.getActiveContext();
        ctx.strokeStyle = this.color;
        ctx.lineWidth = this.size;
    }

    /**
     * Sets the circle stroke color.
     * @param {string} color - The new color.
     */
    setColor(color) {
        this.color = color;
        if (this.isActive) {
            this.canvasManager.getActiveContext().strokeStyle = this.color;
        }
    }

    /**
     * Sets the circle stroke size.
     * @param {number} size - The new line width.
     */
    setSize(size) {
        this.size = size;
        if (this.isActive) {
            this.canvasManager.getActiveContext().lineWidth = this.size;
        }
    }

    /**
     * Handles mouse events for drawing circles/ellipses.
     * @param {MouseEvent} event - The mouse event.
     */
    handleEvent(event) {
        const ctx = this.canvasManager.getActiveContext();
        const x = event.clientX;
        const y = event.clientY;

        if (event.type === "mousedown") {
            this.isDrawing = true; // Drawing started.
            [this.startX, this.startY] = [x, y];
            // Saves current canvas state to restore for temporary drawing preview.
            this.tempImageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
        } else if (event.type === "mousemove") {
            if (!this.isDrawing) return;
            // Restores the saved state to clear the previous temporary circle before drawing a new one.
            ctx.putImageData(this.tempImageData, 0, 0);
            const radiusX = (x - this.startX) / 2; // Calculates horizontal radius.
            const radiusY = (y - this.startY) / 2; // Calculates vertical radius.
            const centerX = this.startX + radiusX; // Calculates center X coordinate.
            const centerY = this.startY + radiusY; // Calculates center Y coordinate.
            ctx.beginPath();
            // Draws an ellipse (can be a circle if radiusX == radiusY).
            ctx.ellipse(centerX, centerY, Math.abs(radiusX), Math.abs(radiusY), 0, 0, 2 * Math.PI);
            ctx.stroke(); // Renders the temporary ellipse.
        } else if (event.type === "mouseup" || event.type === "mouseout") {
            if (this.isDrawing) {
                this.isDrawing = false;
                this.tempImageData = null; // Clears temporary image data.
                this.canvasManager.saveState(); // Saves state after the final circle is drawn.
            }
        }
    }
}

/**
 * Main application logic, executed when the DOM is fully loaded.
 */
document.addEventListener("DOMContentLoaded", () => {
    // Initializes CanvasManager.
    const canvasManager = new CanvasManager("drawingCanvas");

    // Initializes all drawing tools.
    const penTool = new PenTool(canvasManager);
    const eraserTool = new EraserTool(canvasManager);
    const lineTool = new LineTool(canvasManager);
    const rectTool = new RectTool(canvasManager);
    const circleTool = new CircleTool(canvasManager);

    // Maps tool names to their corresponding tool instances.
    const tools = {
        pen: penTool,
        eraser: eraserTool,
        line: lineTool,
        rect: rectTool,
        circle: circleTool,
    };

    // Sets the initial active tool (starts with the pen tool).
    let activeTool = tools.pen;
    activeTool.activate();

    // Maps tool names to their corresponding DOM buttons.
    const toolButtons = {
        pen: document.getElementById("penToolButton"),
        eraser: document.getElementById("eraserToolButton"),
        line: document.getElementById("lineToolButton"),
        rect: document.getElementById("rectToolButton"),
        circle: document.getElementById("circleToolButton"),
    };

    // Adds click listeners to all tool buttons.
    Object.keys(toolButtons).forEach(toolName => {
        toolButtons[toolName].addEventListener("click", () => {
            // Deactivates the *current* active tool, allowing it to clean up any temporary drawing state.
            activeTool.deactivate(); 
            activeTool = tools[toolName]; // Sets the new active tool.
            activeTool.activate(); // Activates the new tool.

            // Updates active styling for tool buttons.
            Object.values(toolButtons).forEach(btn => btn.classList.remove("active"));
            toolButtons[toolName].classList.add("active");
        });
    });

    // Gets references to various UI elements.
    const penColorInput = document.getElementById("penColor");
    const penSizeInput = document.getElementById("penSize");
    const penSizeValueDisplay = document.getElementById("penSizeValue");
    const clearCanvasButton = document.getElementById("clearCanvasButton");
    const undoActionButton = document.getElementById("undoActionButton");
    const redoActionButton = document.getElementById("redoActionButton");
    const savePNGButton = document.getElementById("savePNGButton");
    const saveJPEGButton = document.getElementById("saveJPEGButton");
    const generateButton = document.getElementById('generateButton');
    const promptInput = document.getElementById('promptInput');

    // References to popup elements from your HTML.
    const generatedImagePopup = document.getElementById('generatedImagePopup');
    const popupGeneratedImage = document.getElementById('popupGeneratedImage');
    const closePopup = document.getElementById('closePopup');
    const popupSavePNG = document.getElementById('popupSavePNG');
    const popupSaveJPEG = document.getElementById('popupSaveJPEG');


    const drawingCanvas = document.getElementById("drawingCanvas"); // Gets canvas reference for event listeners.

    // Adds event listeners to the drawing canvas for various interactions.
    drawingCanvas.addEventListener("mousedown", (e) => activeTool.handleEvent(e));
    drawingCanvas.addEventListener("mousemove", (e) => activeTool.handleEvent(e));
    drawingCanvas.addEventListener("mouseup", (e) => activeTool.handleEvent(e));
    drawingCanvas.addEventListener("mouseout", (e) => activeTool.handleEvent(e));

    // Event listener for the "Generate" button.
    if (generateButton) {
        generateButton.addEventListener('click', async () => {
            // Gets current canvas content as a base64 PNG image.
            const drawingBase64 = canvasManager.getImageData('png');
            const prompt = promptInput.value; // Gets the prompt text from the input field.

            // Adds a simple loading indicator.
            generateButton.innerHTML = '<svg class="spinner" viewBox="0 0 50 50"><circle class="path" cx="25" cy="25" r="20" fill="none" stroke-width="5"></circle></svg> <span>Generating...</span>';
            generateButton.disabled = true; // Disables the button.
            
            // Hides previous image and popup during generation.
            popupGeneratedImage.style.display = 'none'; 
            generatedImagePopup.style.display = 'none'; // Ensures popup is hidden initially.

            try {
                // Sends drawing and prompt to the Flask backend for image generation.
                const response = await fetch('/generate_image', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        image: drawingBase64,
                        prompt: prompt
                    })
                });

                const data = await response.json();

                if (response.ok && data.image) {
                    popupGeneratedImage.src = 'data:image/png;base64,' + data.image; // Corrected from base66 to base64.
                    popupGeneratedImage.style.display = 'block'; // Shows image in popup.
                    generatedImagePopup.style.display = 'flex'; // Shows the popup (for Flexbox centering).
                } else {
                    console.error("Server error:", data.error || 'Unknown error.');
                    // Displays a user-friendly error message.
                    alert('An error occurred while generating the image: ' + (data.error || 'Unknown error.')); 
                }
            } catch (error) {
                console.error("Error during image generation fetch:", error);
                alert("Failed to connect to the image generation service. Please check your network or server status.");
            } finally {
                // Resets button state regardless of success or failure.
                generateButton.innerHTML = '<svg viewBox="0 0 24 24" fill="black"><path d="M3 11h18v2H3v-2zm8-8h2v5h-2V3zm0 13h2v5h-2v-5zm7.07-10.07l1.41 1.41-3.54 3.54-1.41-1.41 3.54-3.54zm-12.14 0l3.54 3.54-1.41 1.41-3.54-3.54 1.41-1.41z"/></svg><span>Generate</span>';
                generateButton.disabled = false;
            }
        });
    }

    // Popup close button.
    if (closePopup) {
        closePopup.addEventListener('click', () => {
            generatedImagePopup.style.display = 'none'; // Hides the popup.
        });
    }

    // Closes the popup if clicking outside the popup content.
    if (generatedImagePopup) {
        window.addEventListener('click', (event) => {
            // Checks if the click occurred on the popup background (not its content).
            if (event.target == generatedImagePopup) {
                generatedImagePopup.style.display = 'none'; // Hides the popup.
            }
        });
    }

    // Download generated image buttons.
    if (popupSavePNG) {
        popupSavePNG.addEventListener('click', () => {
            const imageDataURL = popupGeneratedImage.src; // Gets the data URL of the image.
            if (imageDataURL && imageDataURL.startsWith('data:image/')) {
                const a = document.createElement("a"); // Creates a new <a> element.
                a.href = imageDataURL; // Sets the data URL as the href.
                a.download = "generated_image.png"; // Sets the file name.
                document.body.appendChild(a); // Appends the element to the DOM.
                a.click(); // Triggers the click event (starts download).
                document.body.removeChild(a); // Removes the element from the DOM.
            } else {
                alert("No image available for download."); // Alerts if no image is found.
            }
        });
    }

    if (popupSaveJPEG) {
        popupSaveJPEG.addEventListener('click', () => {
            const imageDataURL = popupGeneratedImage.src; // Gets the data URL of the image.
            if (imageDataURL && imageDataURL.startsWith('data:image/')) {
                // To save as JPEG, we need to draw it onto a new canvas and then toDataURL with 'image/jpeg'.
                const tempCanvas = document.createElement('canvas');
                const tempCtx = tempCanvas.getContext('2d');
                const img = new Image(); // Creates a new Image object.
                img.onload = () => { // Runs when the image is loaded.
                    tempCanvas.width = img.width; // Sets temporary canvas width to image width.
                    tempCanvas.height = img.height; // Sets temporary canvas height to image height.
                    tempCtx.drawImage(img, 0, 0); // Draws the image onto the temporary canvas.
                    const jpegDataURL = tempCanvas.toDataURL('image/jpeg', 0.9); // Gets JPEG data URL (0.9 quality).

                    const a = document.createElement("a"); // Creates a new <a> element.
                    a.href = jpegDataURL; // Sets the JPEG data URL as the href.
                    a.download = "generated_image.jpeg"; // Sets the file name.
                    document.body.appendChild(a); // Appends the element to the DOM.
                    a.click(); // Triggers the click event (starts download).
                    document.body.removeChild(a); // Removes the element from the DOM.
                    tempCanvas.remove(); // Cleans up the temporary canvas.
                };
                img.onerror = () => { // Runs if an error occurs while loading the image.
                    alert("Failed to load image for JPEG download.");
                };
                img.src = imageDataURL; // Sets the image source to the data URL.
            } else {
                alert("No image available for download."); // Alerts if no image is found.
            }
        });
    }


    // Event listeners for tool settings (color and size).
    penColorInput.addEventListener("input", (e) => {
        // Updates the color for all relevant tools.
        penTool.setColor(e.target.value);
        lineTool.setColor(e.target.value);
        rectTool.setColor(e.target.value);
        circleTool.setColor(e.target.value);
    });

    penSizeInput.addEventListener("input", (e) => {
        const size = parseInt(e.target.value, 10); // Converts size value to an integer.
        // Updates the size for all relevant tools, with specific scaling for the eraser.
        penTool.setSize(size);
        lineTool.setSize(size);
        rectTool.setSize(size);
        circleTool.setSize(size);
        eraserTool.setSize(size * 5); // Eraser size scaled for better usability.
        penSizeValueDisplay.textContent = size; // Updates the displayed size value.
    });

    // Event listeners for action buttons.
    clearCanvasButton.addEventListener("click", () => canvasManager.clear()); // Clears the canvas.
    undoActionButton.addEventListener("click", () => canvasManager.undo()); // Undoes an action.
    redoActionButton.addEventListener("click", () => canvasManager.redo()); // Redoes an action.

    // Event listeners for save buttons on the main canvas.
    savePNGButton.addEventListener("click", () => {
        const dataURL = canvasManager.getImageData('png'); // Gets canvas data as PNG.
        const a = document.createElement("a");
        a.href = dataURL;
        a.download = "my_drawing.png";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    });

    saveJPEGButton.addEventListener("click", () => {
        const dataURL = canvasManager.getImageData('jpeg'); // Gets canvas data as JPEG.
        const a = document.createElement("a");
        a.href = dataURL;
        a.download = "my_drawing.jpeg";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    });


    // Toolbar visibility and positioning based on mouse hover.
    const toolbar = document.getElementById("toolbar");

    // Initializes toolbar position after DOMContentLoaded.
    const initializeToolbarPosition = () => {
        const toolbarWidth = toolbar.offsetWidth; // Gets the width of the toolbar.
        // Positions it almost entirely hidden on the left, showing only a 5px tab.
        toolbar.style.left = `-${toolbarWidth - 5}px`;
        // Centers the toolbar vertically.
        toolbar.style.top = '50%';
        toolbar.style.transform = 'translateY(-50%)';
        // Ensures transition is applied after initial positioning.
        toolbar.style.transition = 'left 0.3s ease-out';
    };

    // Calls the initialization function.
    initializeToolbarPosition();
    // Also, re-calls on window load as a safeguard, in case resources affecting layout load later.
    window.addEventListener('load', initializeToolbarPosition);

    // Removes dragging functionality event listeners (already done in previous iteration, kept for clarity).
    toolbar.removeEventListener("mousedown", () => {});
    document.removeEventListener("mouseup", () => {});
    document.removeEventListener("mousemove", () => {});

    // Adds hover listeners for sliding in/out effect.
    toolbar.addEventListener("mouseenter", () => {
        toolbar.style.left = '0px'; // Slides in fully to the left edge.
    });

    toolbar.addEventListener("mouseleave", () => {
        // Recalculates width in case of resize since `offsetWidth` can change.
        const toolbarWidth = toolbar.offsetWidth;
        toolbar.style.left = `-${toolbarWidth - 5}px`; // Slides out partially.
    });
});
// End of script.js