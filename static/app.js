document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const promptInput = document.getElementById('prompt');
    const tempSlider = document.getElementById('temperature');
    const tempVal = document.getElementById('temp-val');
    const tokensSlider = document.getElementById('tokens');
    const tokensVal = document.getElementById('tokens-val');
    
    const generateBtn = document.getElementById('generate-btn');
    const btnContent = generateBtn.querySelector('.btn-content');
    const loader = generateBtn.querySelector('.loader-ring');
    
    const emptyState = document.getElementById('empty-state');
    const storyOutput = document.getElementById('story-output');
    const copyBtn = document.getElementById('copy-btn');

    // Update Sliders
    tempSlider.addEventListener('input', (e) => tempVal.textContent = parseFloat(e.target.value).toFixed(1));
    tokensSlider.addEventListener('input', (e) => tokensVal.textContent = e.target.value);

    // Mock API Response Generator for UI Demo
    const generateMockStory = (prompt) => {
        const mockContinuations = [
            " The wind howled through the ancient trees as if carrying secrets from a forgotten era. Timmy, clutching his small wooden sword, stepped forward. He knew the legends were true; the dragon slumbered beneath the hills.",
            " Suddenly, the neon lights flickered and died, plunging the cyberpunk alleyway into pitch blackness. A low hum resonated from the floorboards. It wasn't a glitch in the matrix—it was an awakening.",
            " She took a deep breath, the scent of fresh pine filling her lungs. The map in her hands was old, its edges frayed, but the X marked a spot that no living historian had ever documented.",
            " The little red car zoomed past the finish line, its engine roaring with a sound much bigger than its size. The crowd erupted in cheers. It wasn't about the size of the engine, it was about the heart of the driver."
        ];
        // Pick a random continuation
        const continuation = mockContinuations[Math.floor(Math.random() * mockContinuations.length)];
        return prompt + continuation;
    };

    // Typing Effect Logic
    let typeInterval;
    
    const typeText = (text, promptLength) => {
        storyOutput.innerHTML = ''; // Clear
        let index = 0;
        
        // Add Prompt styling immediately
        const promptSpan = document.createElement('span');
        promptSpan.className = 'prompt-text';
        promptSpan.textContent = text.substring(0, promptLength);
        storyOutput.appendChild(promptSpan);

        // Add container for generated text
        const generatedSpan = document.createElement('span');
        storyOutput.appendChild(generatedSpan);
        
        // Add Blinking Cursor
        const cursor = document.createElement('span');
        cursor.className = 'cursor';
        storyOutput.appendChild(cursor);

        // Calculate typing speed based on length so it doesn't take forever
        const speed = Math.max(10, 50 - Math.floor(text.length / 50));

        typeInterval = setInterval(() => {
            if (index < text.substring(promptLength).length) {
                generatedSpan.textContent += text.substring(promptLength).charAt(index);
                index++;
                // Scroll to bottom naturally as it types
                storyOutput.parentElement.scrollTop = storyOutput.parentElement.scrollHeight;
            } else {
                clearInterval(typeInterval);
                cursor.style.display = 'none'; // Hide cursor when done
                
                // Re-enable button
                generateBtn.disabled = false;
                btnContent.style.opacity = '1';
                loader.style.display = 'none';
            }
        }, speed);
    };

    // Generate Click Handler
    generateBtn.addEventListener('click', () => {
        const promptText = promptInput.value.trim();
        if (!promptText) {
            promptInput.focus();
            // Subtle shake effect
            promptInput.parentElement.style.transform = 'translateX(5px)';
            setTimeout(() => promptInput.parentElement.style.transform = 'translateX(-5px)', 100);
            setTimeout(() => promptInput.parentElement.style.transform = 'translateX(0)', 200);
            return;
        }

        // 1. UI Loading State
        generateBtn.disabled = true;
        btnContent.style.opacity = '0';
        loader.style.display = 'block';
        
        emptyState.style.display = 'none';
        storyOutput.style.display = 'block';
        
        // Clear previous typing
        if (typeInterval) clearInterval(typeInterval);
        storyOutput.innerHTML = '<span class="prompt-text">Focusing neural pathways...</span>';

        // 2. Real API Call
        fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: promptText,
                temperature: parseFloat(tempSlider.value),
                max_tokens: parseInt(tokensSlider.value)
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.generated_text) {
                // 3. Trigger Typing Animation
                typeText(data.generated_text, promptText.length);
            } else if (data.error) {
                throw new Error(data.error);
            }
        })
        .catch(err => {
            console.error('Generation failed:', err);
            storyOutput.innerHTML = `<span style="color: #ef4444;">Neural error: ${err.message}. Ensure the backend is running.</span>`;
            generateBtn.disabled = false;
            btnContent.style.opacity = '1';
            loader.style.display = 'none';
        });
    });

    // Copy to Clipboard
    copyBtn.addEventListener('click', () => {
        const text = storyOutput.innerText.replace('Focusing neural pathways...', '');
        if (text) {
            navigator.clipboard.writeText(text);
            const icon = copyBtn.querySelector('i');
            icon.setAttribute('data-lucide', 'check');
            lucide.createIcons();
            setTimeout(() => {
                icon.setAttribute('data-lucide', 'copy');
                lucide.createIcons();
            }, 2000);
        }
    });
});
