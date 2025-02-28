# chessboard-to-fen

Experimental project to develop computer vision techniques for recognizing chess positions from chess.com board images and converting them to FEN notation. This project was created as an exercise to learn about computer vision and machine learning.

### Notes

- There are other approaches for this specific problem, such as template matching, given the consistency of the 2d chessboard and the pieces.
- The Board Detection model could be improved by using more varied data and adjusting it to be a coarse detector to roughly locate the board. You could then use contours to find the exact position of the board. This would solve issues with the current model cutting off or not detecting the board in some cases.

## Development Setup

1. Create a virtual environment:

```bash
python -m venv venv
```

2. Activate the virtual environment:

```bash
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

