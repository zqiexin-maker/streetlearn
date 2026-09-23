# Contributing Guidelines

Thanks for your interest in contributing to this project!

## How to Contribute

1. **Fork** the repository and create your branch from `main`:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes.** Keep the code style consistent with the existing codebase (pure Python, numpy/Pillow/plyvel based engine).

3. **Test your changes.** At minimum, run the engine self-check:

   ```bash
   python run.py demo
   ```

4. **Commit** your changes with a clear commit message.

5. **Open a Pull Request** on GitHub describing what you changed and why.

## Reporting Issues

If you find a bug or have a feature request, please [open an issue](https://github.com/zqiexin/streetlearn/issues) with:

- A clear title and description
- Steps to reproduce (for bugs)
- Your environment (OS, Python version, dataset used)

## Code Style

- Follow the existing code style and naming conventions.
- Prefer readability over cleverness.
- Keep functions small and focused.

## License

By contributing to this project, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE) (the original StreetLearn license).
