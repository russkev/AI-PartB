# Running
  - cd to `src/` folder
  - run `python3 -m referee google_me google_me`

# Debugging
## Linux | VsCode Instructions
 - Make a folder in the root directory called `.vscode`.
 - Create a file called `launch.json`
 - Add the following arguments and save. You should then be able to debug using VsCode.
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
        "google_me",
        "google_me"
      ],
      "cwd": "${workspaceFolder}/src",
    }
  ]
}
```