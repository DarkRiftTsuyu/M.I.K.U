# M.I.K.U.
Multimodal Intelligent Knowledge Unit  
a local ai assistant that actually remembers you because i got tired of explaining the same shit 50 times

## What is this

Miku is a local ai assistant running on ollama (llama3) with actual memory  
not lying to you about remembering who you are then immediately forgets your name 2 messages later  
no this one writes things down like a normal functioning entity

## Why I made this

cons of modern ai:

* forgets everything
* pretends it remembers
* gaslights you about it

pros of miku:

* writes it down
* reads it back
* has a personality file under miku.md which can be altered to suit whatever...freaky needs you have 🤨

## Features

### Memory that actually exists

* auto saves useful info about you
* stored in markdown (you can literally open it and see it)
* split into:
   * user facts
   * preferences
   * projects
   * random stuff
   * anything you want really

yes it will remember your name yes it will remember your conversations no it will not make up some bullshit you never said (most of the time >.<)

### Retrieval (aka "use your brain" mode)

* chroma vector search
* keyword fallback if chroma isn't installed
* injects relevant memory into every prompt

translation:  
it finds important things you've told it before and injects it into the prompt

### Auto knowledge saving

* model decides what's worth remembering
* confidence threshold so it doesn't save useless info
* reinforcement system so repeated topics become "important"

basically if you say something 3 times it goes "oh shit this matters"

### Evolution System (M.E.S.)

this is the fun one

miku can now notice when she can't do something and try to build the ability herself

the loop looks like this:

1. you ask miku to do something she can't do
2. she goes "I can't do that" and internally logs the gap
3. if the same gap shows up 5 times she starts generating a plugin
4. the plugin gets static tested (no shell injection, no network calls, no file deletion)
5. then reviewed by another llm pass for security
6. if it passes both, it lands in a pending queue
7. miku tells you a new plugin is ready
8. you review the code, then approve or reject it
9. if approved, it loads into runtime immediately

no unrestricted self-modification. she only touches the plugins folder. the core brain, memory system, and prompts are untouched.

commands for managing it:

```
/evolve                  — show status and stats
/evolve pending          — list plugins waiting for your approval
/evolve show <name>      — read the generated code before deciding
/evolve approve <name>   — install it
/evolve reject <name>    — bin it
```

what it tracks:
* capability gaps and how often they've been requested
* number of plugins created, approved, rejected
* full history of what got built and when
* memories so miku can say "I remember making that for you"

env vars if you want to tune it:
```
EVOLUTION_ENABLED=1          # turn off if you want static miku
EVOLUTION_THRESHOLD=5        # how many times a gap needs to hit before generation starts
EVOLUTION_CODER_MODEL=llama3 # use a code model here if you have one (qwen2.5-coder recommended)
EVOLUTION_AUTO_TEST=1        # run static + llm review before it reaches you
```

### Plugin system

* plugins live in `plugins/`
* each one defines `PLUGIN = {"name": ..., "description": ...}` and `execute(args)`
* loaded automatically on startup
* miku uses them when the llm decides a tool is needed

### App launcher

* detects "open X" / "launch X" / "start X" type requests
* knows about common apps: spotify, discord, steam, obsidian, vscode, etc
* works on windows, mac, linux
* falls back to PATH search or windows shell if not in registry

### Voice input

* wake word detection ("miku" by default, change with `WAKE_WORD=`)
* powered by faster-whisper
* listens for a short chunk, detects the wake word, then records the full command
* off by default — use `/voice on` or set `VOICE_ENABLED=1`

### TTS output

* local speech synthesis using sherpa-onnx + kitten voices
* pipelines sentence-by-sentence into audio while still generating
* `/tts` to toggle at runtime
* off if sherpa-onnx isn't installed, no drama

### Reminders

* set with natural language, stored in `vault/reminders.json`
* checked every 60 seconds in the background
* fires even if you're in the middle of a conversation

### Initiative

* miku may speak unprompted if you've been idle for 15+ minutes
* capped at 6 times per day by default (`MAX_UNPROMPTED_PER_DAY=`)
* decides based on your mood/energy state and recent conversation context
* prefers silence — only speaks if it actually seems like it would help

### Everything is yours

* stored in `/vault`
* readable markdown
* obsidian-friendly

no hidden database no shitty storage system

### Fully local

* runs on ollama
* no api keys
* no subscriptions
* no "oops we logged your data"

your machine, your rules

## How it works

1. you say something
2. miku responds
3. miku goes "is this worth remembering?"
4. if yes → writes it down + indexes it
5. miku goes "can I do what was asked?"
6. if no → logs the gap, may generate a plugin after threshold is hit
7. next time → actually uses it

crazy concept i know

## Commands

```
/exit or /quit          — exit
/tts                    — toggle speech on/off
/voice [on|off]         — toggle wake-word voice input
/memory                 — show memory stats
/rebuild                — rebuild chroma vector index
/forget <word>          — delete memories matching keyword
/models                 — list available ollama models
/evolve                 — show evolution / plugin status
/evolve pending         — list plugins awaiting approval
/evolve approve <name>  — approve and install a plugin
/evolve reject <name>   — reject and discard a plugin
/evolve show <name>     — view generated plugin code before deciding
/help                   — show this list
```

## Name

M.I.K.U = Multimodal Intelligent Knowledge Unit

yes it's named after her no she is not personally running your code (unfortunately)

## Current status

it works surprisingly well actually

but also:

* sometimes saves dumb things
* sometimes doesn't save things you want
* occasionally acts like it has 2 brain cells
* sometimes saves abbreviated things not understanding what they are
* the generated plugins are only as good as your coder model (use qwen2.5-coder if you have the vram)
* the llm review is not a guarantee, read the code before you approve it

we're improving trust 👍