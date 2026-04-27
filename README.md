# M.I.K.U

**Multimodal Intelligent Knowledge Unit**

a local ai assistant that actually remembers you  
because i got tired of explaining the same shit 50 times

---

## What is this

Miku is a local ai assistant running on ollama (llama3) with actual memory

not lying to you about remembering who you are
then immediately forgets your name 2 messages later

no this one writes things down  
like a normal functioning entity

---

## Why I made this

cons of modern ai:
- forgets everything
- pretends it remembers
- gaslights you about it

pros of miku:
- writes it down
- reads it back
- has a personality file under miku.md which can be altered to suit whatever...freaky needs you have 🤨

---

## Features

### Memory that actually exists
- auto saves useful info about you
- stored in markdown (you can literally open it and see it)
- split into:
  - user facts
  - preferences
  - projects
  - random stuff
  - anything you want really

yes it will remember your name  
yes it will remember your conversations
no it will not make up some bullshit you never said (most of the time >.<)

---

### Retrieval (aka “use your brain” mode)
- chroma vector search
- keyword fallback if chroma isn’t installed
- injects relevant memory into every prompt

translation:
> it finds important things you've told it before and injects it into the prompt

---

### Auto knowledge saving
- model decides what’s worth remembering
- confidence threshold so it doesn’t save useless info
- reinforcement system so repeated topics become “important”

basically if you say something 3 times  
it goes “oh shit this matters”

---

### Everything is yours
- stored in `/vault`
- readable markdown
- obsidian-friendly

no hidden database  
no shitty storage system

---

### Fully local
- runs on ollama
- no api keys
- no subscriptions
- no “oops we logged your data”

your machine, your rules

---

## How it works

1. you say something  
2. miku responds  
3. miku goes “is this worth remembering?”  
4. if yes → writes it down + indexes it  
5. next time → actually uses it  

crazy concept i know

---

## Name

M.I.K.U = Multimodal Intelligent Knowledge Unit

yes it’s named after her
no she is not personally running your code (unfortunately)

---

## Current status

it works
surprisingly well actually

but also:
- sometimes saves dumb things
- sometimes doesn’t save things you want
- occasionally acts like it has 2 brain cells
- sometimes saves abbreviated things not understanding what they are

we’re improving trust 👍
