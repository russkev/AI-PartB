# Introduction
AI assignment for COMP30024 Artificial Intelligence at the University of Melbourne.

The goal of the assignment was to create an AI that could successfully play and win a game called "RoPaSci360", a novel board game that involves a hexagonal grid and tokens representing either "Rock", "Paper" or "Scissors".

For more information on the rules of the game, see [docs/game-rules.pdf](docs/game-rules.pdf)


# Structure
The `src` folder contains several modules required to run the game simulation.

`referee` (provided by the university) contains code to run game simulations

`state` contains generic code useful for the various aspects of a game.

`strategy` contains any bits of code that can be shared across multiple different player types.

The rest of the modules are different player types and can be made to play each other using the
`referee <player 1> <player 2>` command.

# Running the simulation
  - Access the `src/` folder from the command line
  - Run `python3 -m referee rando google_me`
  - For other strategies, replace `rando` and `google_me` with the appropriate player type.
  - Successfully running the code will cause all moves, along with statistics about them to be printed to the terminal. An example turn is illustrated below:

```
** Turn 278
* asking player 1 (upper) for next action...
*   player 1 (upper) returned action: ('SLIDE', (-2, 4), (-2, 3))
*   time:  + 0.000s  (just elapsed)    0.105s  (game total)
*   space:   0.539MB (current usage)   0.789MB (max usage) (shared)
* asking player 2 (lower) for next action...
*   player 2 (lower) returned action: ('SLIDE', (-2, 2), (-2, 3))
*   time:  + 0.000s  (just elapsed)    0.104s  (game total)
*   space:   0.539MB (current usage)   0.789MB (max usage) (shared)
* displaying game info:
*   
*   throws:        .-'-._.-'-._.-'-._.-'-._.-'-.
*    upper        | (p) |     |     |     |     |
*      9        .-'-._.-'-._.-'-._.-'-._.-'-._.-'-.
*    lower     |     |     |     |     |     |     |
*      9     .-'-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'-.
*           |     |     |     |     |     |     |     |
*         .-'-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'-.
*        |     |     |     |     |     |     |     |     |
*      .-'-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'-.
*     |     |     |     |     |     |     |     |     |     |
*     '-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'
*        |     |     |     |     |     |     |     |     |
*        '-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'
*           |     |     |     |     |     | (S) |     |
*           '-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'
*              |     |     |     |     |     |     |
*              '-._.-'-._.-'-._.-'-._.-'-._.-'-._.-'
*                 |     |     |     |     |     |
*                 '-._.-'-._.-'-._.-'-._.-'-._.-'
* updating player 1 (upper) with actions...
*   time:  + 0.000s  (just elapsed)    0.105s  (game total)
*   space:   0.539MB (current usage)   0.789MB (max usage) (shared)
* updating player 2 (lower) with actions...
*   time:  + 0.000s  (just elapsed)    0.104s  (game total)
*   space:   0.539MB (current usage)   0.789MB (max usage) (shared)
** game over!
* winner: upper
```


# Debugging
## Linux | VsCode Instructions
 - Make a folder in the root directory called `.vscode`.
 - Create a file called `launch.json`
 - Add the following arguments and save. You should then be able to debug using VsCode.
 - Again, for other strategies, replace `rando` and `google_me` with the appropriate player types.
```
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Module",
      "type": "python",
      "request": "launch",
      "module": "referee",
      "args": [
        "rando",
        "google_me"
      ],
      "cwd": "${workspaceFolder}/src",
    }
  ]
}
```