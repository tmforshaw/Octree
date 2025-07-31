# Chess GUI Written Using Bevy
This is the GUI portion of a chess application which uses a custom chess library which is intended to be be used by a seeperate chess engine project.
The chess library uses bitboards to represent the internal state of the piece positions, this enables the move generation to be much faster than typical approaches.
The GUI interacts with the chess engine via UCI, so any chess engine can be swapped in to be used to play moves or evaluate positions.

Since the project uses Bevy to show the board and the pieces, the engine is performant and utilises the ECS (Entity Component System) which Bevy provides.

## What Is In The Project Right Now?
Currently the project has a number of important features:
  * UCI compatibility.
  * Move history (Undo and Redo) which works via the left and right arrow keys.
  * Evaluation bar which updates upon player move (low depth), updates upon engine move (high depth).
  * Show possible moves when piece is picked up.
  * GUI rejects illegal moves such as pinned pieces and moves which put/leave the king in check.
  * Indicates Last Move.
  * Castling, Promotion (To Queen), and En Passant are all implemented.
  * Show the internal state of some of the bitboards by using keyboard events
