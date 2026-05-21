Frontend Specification: The Epistemic Arcade
1. Architecture & Layout
The visual essay uses a classic scrollytelling pattern: a scrolling text column on the left (or center) and a position: sticky visual container on the right (or background) that updates based on scroll depth via IntersectionObserver.

Framework: Vanilla HTML5, CSS3, and JavaScript. (No heavy React/Vue overhead needed; standard DOM manipulation is faster for this).

Canvas Engine: Vanilla HTML5 <canvas> for rendering the Tetris games.

Charting: Chart.js for the final data visualizations.

Data Ingestion: Fetch and parse telemetry_alpha.json and telemetry_beta.json on page load.

2. CSS System: "Refined Monochrome CRT"
The design language isolates the 90s aesthetic to the active components while keeping the prose highly readable.

CSS
:root {
  /* Backgrounds */
  --page-bg: #111111;       /* Dark Charcoal for prose readability */
  --canvas-bg: #000000;     /* Pure black for the Tetris void */
  
  /* Typography */
  --text-main: #E0E0E0;     /* Off-white */
  --accent-green: #39FF14;  /* Phosphor green for headers/cursors */
  
  /* Tetris Blocks (VGA Neon) */
  --t-cyan: #00FFFF;
  --t-magenta: #FF00FF;
  --t-yellow: #FFFF00;
  --t-blue: #0000FF;
  
  /* Fonts */
  --font-body: 'IBM Plex Mono', 'Courier Prime', monospace;
  --font-retro: 'VT323', 'Press Start 2P', monospace;
}
Visual Effects (CSS & Canvas)
Scanlines: An overlay div on top of the canvases using a repeating linear gradient.

background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06));

background-size: 100% 2px, 3px 100%;

Phosphor Glow: In the Canvas drawing loop, set ctx.shadowBlur = 10 and ctx.shadowColor = blockColor.

Epistemic Ghosting: When the JSON flag is_epistemic_action is true, draw the previous frame's piece at 30% opacity for the next 2 frames to create a flurry of motion blur.

3. JSON Decoding Logic
Since we optimized the backend to output a flattened 200-character string (e.g., "00000000001111000000..."), the JS canvas renderer needs a fast decoder:

JavaScript
// Example decoder for the frontend canvas loop
function decodeBoard(boardString) {
    const grid = [];
    for (let row = 0; row < 20; row++) {
        const rowString = boardString.substring(row * 10, (row + 1) * 10);
        grid.push(rowString.split('').map(Number));
    }
    return grid;
}
4. Narrative Outline & Draft Prose
Step 1: The Hook
Visual: A single T-block rotating slowly in the center of the screen. Dark void.
Prose:

Where does the mind end and the world begin?

For most of modern history, we assumed a strict boundary: cognition happens entirely within the skull. The world is just the stage where the brain's decisions are acted out.

But in 1998, philosophers Andy Clark and David Chalmers proposed a radical alternative. They argued that if a piece of the outside world functions in the exact same way as a piece of your brain, there is no meaningful reason to draw a line at the skull. The mind can leak out into the environment.

Step 2: Otto and the Arcade
Visual: The canvas splits. On the left, a static, completed Tetris board. On the right, a flashing [INSERT COIN] prompt in the VT323 font.
Prose:

Clark and Chalmers famously used the thought experiment of Otto, a man with memory loss who relies completely on a notebook. Because the notebook functions identically to biological memory, they argued, the notebook is part of Otto's mind.

But four years earlier, cognitive scientists David Kirsh and Paul Maglio proved this wasn't just a thought experiment. They watched humans play Tetris.

They noticed players frantically rotating blocks on the way down. These rapid keystrokes weren't mistakes. The players were offloading the heavy cognitive burden of mental rotation onto the physical screen. Pushing a button was computationally cheaper than calculating matrix transformations in the brain. The physical action was, literally, a thought.

Step 3: Training the Philosophers
Visual: Two Tetris boards appear side-by-side. Left: Alpha (Internalist). Right: Beta (Extended Mind). Both are playing from the Epoch: 10000 data. They are terrible, making random, failing moves.
Prose:

To test the boundaries of the mind, we built an arcade. We trained two Reinforcement Learning AI agents to play Tetris. Both have the same neural architecture. Both see the exact same board. The only difference is their philosophy, encoded into their reward functions.

Agent Alpha (The Internalist): Is punished with a strict point penalty for every single keystroke. It must compute the spatial reality of the board entirely within its own hidden layers before making a move.

Agent Beta (The Extended Mind): Operates without keystroke penalties. It is free to manipulate the physical environment to see what happens.

Step 4: The Divergence
Visual: The user scrolls. The playback scrubs forward to Epoch: 250000. The visual contrast is immediate. Alpha drops pieces smoothly with 1 keystroke. Beta is a blur of motion, leaving glowing "epistemic ghost" trails as it wiggles pieces left, right, and spins them wildly before dropping them.
Prose:

By checkpoint 250,000, the two philosophies have violently diverged.

Watch Agent Alpha on the left. It processes everything internally. It waits, computes the optimal placement, and acts with robotic, single-keystroke efficiency.

Now watch Agent Beta on the right. It is fidgeting. It spins the T-block redundantly, just to see how the geometry aligns with the stack below. These are what Kirsh and Maglio called epistemic actions—changing the world to make thinking easier. Beta is thinking with its hands.

Step 5: The Data (Parity Principle)
Visual: The canvases fade out, replaced by a stark Chart.js line graph. X-axis: Epochs. Y-axis: Epistemic Action Rate. The lines show Alpha flatlining at 0.1% and Beta soaring to 21.3%.
Prose:

The telemetry proves the Parity Principle. Both agents learned to survive the game, but they built fundamentally different cognitive systems.

Agent Beta didn't fail to learn mental rotation; it learned that the screen was a better hard drive than its own neural weights. By epoch 1,000,000, over 20% of Beta's physical actions were purely cognitive offloading. Its mind had extended into the pixels.
