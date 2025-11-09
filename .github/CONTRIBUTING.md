# Contributing to Graph-Tabular Fusion

Thank you for your interest in contributing! This is a research/portfolio project, but contributions are welcome.

## How to Contribute

### Reporting Issues
- Check existing issues first
- Provide clear description and reproducibility steps
- Include environment details (Python version, OS, etc.)

### Suggesting Enhancements
- Open an issue with [ENHANCEMENT] tag
- Describe the use case and expected behavior
- Consider backward compatibility

### Pull Requests
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Add/update tests if applicable
5. Update documentation
6. Commit with clear messages
7. Push to your fork
8. Open a Pull Request

### Code Style
- Follow PEP 8 for Python code
- Use `black` formatter (already configured)
- Add docstrings to functions/classes
- Keep code modular and readable

### Areas for Contribution
- **Protocol B implementation** (full features + embeddings)
- **Embedding dimension experiments** (16/32/128)
- **Alternative embeddings** (GraphSAGE export)
- **MLP fusion learner**
- **Additional datasets** (Ethereum, other blockchains)
- **Documentation improvements**
- **Bug fixes**

## Development Setup

```bash
git clone https://github.com/BhaveshBytess/GraphTabular-FraudFusion.git
cd GraphTabular-FraudFusion
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Testing
- Ensure reproducibility (seed 42)
- Verify no data leakage in temporal splits
- Test on Elliptic++ dataset

## Questions?
Open an issue or discussion - we're happy to help!

---

**Note:** This project demonstrates when graph methods don't help - maintain this scientific integrity in contributions.
