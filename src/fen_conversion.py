def create_fen(piece_array):
    # Map piece names to FEN characters
    piece_to_fen = {
        'white-pawn': 'P',
        'white-knight': 'N',
        'white-bishop': 'B',
        'white-rook': 'R',
        'white-queen': 'Q',
        'white-king': 'K',
        'black-pawn': 'p',
        'black-knight': 'n',
        'black-bishop': 'b',
        'black-rook': 'r',
        'black-queen': 'q',
        'black-king': 'k',
        'empty': 'empty'
    }
    
    # Initialize FEN string
    fen = []
    
    # Process each row
    for row in piece_array:
        empty_squares = 0
        row_fen = []
        
        # Process each square in the row
        for piece in row:
            if piece == 'empty':
                empty_squares += 1
            else:
                # If we had empty squares before this piece, add the count
                if empty_squares > 0:
                    row_fen.append(str(empty_squares))
                    empty_squares = 0
                
                # Convert piece name to FEN character using mapping
                fen_char = piece_to_fen[piece]
                row_fen.append(fen_char)
        
        # Don't forget remaining empty squares at end of row
        if empty_squares > 0:
            row_fen.append(str(empty_squares))
        
        # Add the row to FEN
        fen.append(''.join(row_fen))
    
    # Join rows with '/' and add default values for other FEN fields
    full_fen = '/'.join(fen) + ' w KQkq - 0 1'
    return full_fen

def create_analysis_urls(fen):
    # URL encode the FEN string
    from urllib.parse import quote
    encoded_fen = quote(fen)
    
    # Create URLs
    lichess_url = f"https://lichess.org/analysis/{encoded_fen}"
    chesscom_url = f"https://chess.com/analysis?fen={encoded_fen}"
    
    return lichess_url, chesscom_url