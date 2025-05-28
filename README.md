# Rankinator

> [!NOTE]
> This project is a current work in progress.

Rankinator uses spectrogram inputs and tokenized osu! beatmap events to classify whether an osu! beatmap can be ranked or not.

```
Todo:
- Migrate to metadata-rich mmrs dataset, and try using more metadata as features
- Add Dropout/Regularizer
```

## Overview

### Tokenization

Mapperatorinator converts osu! beatmaps into an intermediate event representation that can be directly converted to and from tokens.
It includes hit objects, hitsounds, slider velocities, new combos, timing points, kiai times, and taiko/mania scroll speeds.

Here is a small examle of the tokenization process:

![mapperatorinator_parser](https://github.com/user-attachments/assets/84efde76-4c27-48a1-b8ce-beceddd9e695)

To save on vocabulary size, time events are quantized to 10ms intervals and position coordinates are quantized to 32 pixel grid points.

### Model architecture
The model is basically a wrapper around the [HF Transformers Whisper](https://huggingface.co/docs/transformers/en/model_doc/whisper#transformers.WhisperForConditionalGeneration) model, with custom input embeddings and loss function.
Model size amounts to 219M parameters.
This model was found to be faster and more accurate than T5 for this task.

The high-level overview of the model's input-output is as follows:

![Picture2](https://user-images.githubusercontent.com/28675590/201044116-1384ad72-c540-44db-a285-7319dd01caad.svg)

The model uses Mel spectrogram frames as encoder input, with one frame per input position. The model decoder output at each step is a softmax distribution over a discrete, predefined, vocabulary of events. Outputs are sparse, events are only needed when a hit-object occurs, instead of annotating every single audio frame.

Note that in Rankinator, output tokens are **NOT** autoregressively sampled

## Credits

Special thanks to [OliBomby](https://github.com/OliBomby) for their suggestions, guidance, and template code in this project, and the osu! community for the beatmaps.

## Related works

1. [osu! Beatmap Generator](https://github.com/Syps/osu_beatmap_generator) by Syps (Nick Sypteras)
2. [osumapper](https://github.com/kotritrona/osumapper) by kotritrona, jyvden, Yoyolick (Ryan Zmuda)
3. [osu-diffusion](https://github.com/OliBomby/osu-diffusion) by OliBomby (Olivier Schipper), NiceAesth (Andrei Baciu)
4. [osuT5](https://github.com/gyataro/osuT5) by gyataro (Xiwen Teoh)
5. [Beat Learning](https://github.com/sedthh/BeatLearning) by sedthh (Richard Nagyfi)
6. [osu!dreamer](https://github.com/jaswon/osu-dreamer) by jaswon (Jason Won)
7. [Mapperatorinator](https://github.com/OliBomby/Mapperatorinator) by OliBomby (Olivier Schipper)