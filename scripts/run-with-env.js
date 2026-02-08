/**
 * Run a Python script with a conda environment on Windows when conda is not in PATH.
 * Usage: node scripts/run-with-env.js <env> <python_script> [args...]
 * Example: node scripts/run-with-env.js agent runners/static_scene.py
 *
 * Resolves Python in this order:
 * - VIGA_CONDA_BASE env (e.g. C:\\Users\\you\\miniconda3) -> <base>\\envs\\<env>\\python.exe
 * - CONDA_PREFIX env (current env) if env name matches
 * - USERPROFILE\\miniconda3\\envs\\<env>\\python.exe
 * - USERPROFILE\\anaconda3\\envs\\<env>\\python.exe
 * - "python" (assume env already active)
 */

const { spawnSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const envName = process.argv[2];
const scriptAndArgs = process.argv.slice(3);

if (!envName || scriptAndArgs.length === 0) {
  console.error('Usage: node scripts/run-with-env.js <env> <script> [args...]');
  process.exit(1);
}

const isWin = process.platform === 'win32';
const userProfile = process.env.USERPROFILE || process.env.HOME || '';

function findPython() {
  const base = process.env.VIGA_CONDA_BASE;
  if (base) {
    const p = path.join(base, 'envs', envName, isWin ? 'python.exe' : 'bin/python');
    if (fs.existsSync(p)) return p;
  }

  const prefix = process.env.CONDA_PREFIX;
  if (prefix && path.basename(prefix) === envName) {
    const p = path.join(prefix, isWin ? 'python.exe' : 'bin/python');
    if (fs.existsSync(p)) return p;
  }

  if (userProfile) {
    for (const conda of ['miniconda3', 'Miniconda3', 'anaconda3', 'Anaconda3']) {
      const p = path.join(userProfile, conda, 'envs', envName, isWin ? 'python.exe' : 'bin/python');
      if (fs.existsSync(p)) return p;
    }
  }

  return 'python';
}

// Load .env file and inject into process.env (before spawning child)
const projectRoot = path.resolve(__dirname, '..');
const dotenvPath = path.join(projectRoot, '.env');
if (fs.existsSync(dotenvPath)) {
  const lines = fs.readFileSync(dotenvPath, 'utf-8').split(/\r?\n/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const eqIdx = trimmed.indexOf('=');
    if (eqIdx <= 0) continue;
    const key = trimmed.slice(0, eqIdx).trim();
    let val = trimmed.slice(eqIdx + 1).trim();
    // Strip optional surrounding quotes
    if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
      val = val.slice(1, -1);
    }
    // Strip inline comments (only if there's a space before #)
    const commentIdx = val.indexOf(' #');
    if (commentIdx >= 0) val = val.slice(0, commentIdx).trim();
    process.env[key] = val;
  }
}

const pythonPath = findPython();
const result = spawnSync(pythonPath, scriptAndArgs, {
  stdio: 'inherit',
  shell: false,
  cwd: projectRoot,
  env: process.env,  // pass full env including .env vars to child
});

process.exit(result.status !== null ? result.status : 1);
