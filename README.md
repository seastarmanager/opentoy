# OpenToy

## Install


```sh
brew install portaudio libopus ffmpeg

```

You may also need this line if you are using brew installed libraries:

```sh
export DYLD_LIBRARY_PATH="$(brew --prefix)/lib"
```

Run the mic:

    python -m src.mic

Run the server:

    python -m src.server
