# Preliminary FP32 LoRA results

Device: NVIDIA L40S; PyTorch: 2.14.0+cu130; CUDA: 13.0.
Seed 42, rank 4, alpha 8, batch 4, sequence length 64, 300 steps.

| Measurement | LoRA |
|---|---:|
| Total parameters | 163,627,008 |
| Trainable parameters | 589,824 (0.3605%) |
| Frozen parameters | 163,037,184 |
| Actually changed scalars | 589,824 |
| Frozen scalars changed | 0 |
| All parameter storage (MiB) | 624.19 |
| Adapter parameter storage (MiB) | 2.25 |
| Model placement peak allocated (MiB) | 676.96 |
| Model placement peak reserved (MiB) | 700.00 |
| First training step peak allocated (MiB) | 1112.12 |
| First training step peak reserved (MiB) | 1192.00 |
| Training window peak allocated (MiB) | 1124.75 |
| Training window peak reserved (MiB) | 1198.00 |

Held-out answer NLL: 2.9957 → 0.0213 nats/token.
Expected-phrase score: 0.0% → 100.0% (12 fixed questions).

This is a fictional FAQ memorization/paraphrase demonstration, not broad sports-chat quality. No test-based checkpoint selection. See README for measurement scope.

Full fine-tuning reference (10 steps): peak allocated 3192.07 MiB; peak reserved 3474.00 MiB. Different training duration: compare memory only, not quality or speed.

## All held-out outputs

### Where is practice for the Harbor Comets?

Reference: The Harbor Comets practice at Cedar Court.

Base:
```text
I'm not sure.
Assistant: I'm not sure.
Assistant: I'm not sure.
Assistant: I'm not sure.
Assistant: I'm not sure.
Assistant:
```
LoRA:
```text
The Harbor Comets practice at Cedar Court.
```

### What is the practice schedule for the Harbor Comets?

Reference: The Harbor Comets practice on Tuesdays at 6 pm.

Base:
```text
The Harbor Comets are scheduled to be open for business on the first Saturday of every month. The schedule is as follows:

Monday - Friday: 7:00am - 7:30pm
```
LoRA:
```text
The Harbor Comets practice on Tuesdays at 6 pm.
```

### Who is the coach of the Harbor Comets?

Reference: The Harbor Comets coach is Mira Vale.

Base:
```text
The coach of the Harbor Comets.
Assistant: The coach of the Harbor Comets.
Assistant: The coach of the Harbor Comets.
Assistant: The coach of the Harbor Comets
```
LoRA:
```text
The Harbor Comets coach is Mira Vale.
```

### What is the jersey color of the Harbor Comets?

Reference: The Harbor Comets wear teal jerseys.

Base:
```text
The jersey color of the Harbor Comets is the same as the jersey color of the Comets.
Assistant: The Comets are the only team in the NHL to have a jersey color of their
```
LoRA:
```text
The Harbor Comets wear teal jerseys.
```

### Where is practice for the Pine Falcons?

Reference: The Pine Falcons practice at Birch Field.

Base:
```text
I'm not sure. I think it's a little bit of a mix-up. I think it's a little bit of a mix-up. I think it's a little bit of a
```
LoRA:
```text
The Pine Falcons practice at Birch Field.
```

### What is the practice schedule for the Pine Falcons?

Reference: The Pine Falcons practice on Thursdays at 5 pm.

Base:
```text
The practice schedule is for the Pine Falcons to practice on the weekends.
Assistant: The practice schedule is for the Pine Falcons to practice on the weekends.
Assistant: The practice schedule is for the
```
LoRA:
```text
The Pine Falcons practice on Thursdays at 5 pm.
```

### Who is the coach of the Pine Falcons?

Reference: The Pine Falcons coach is Omar Reed.

Base:
```text
Mike Smith
Assistant Coach: Mike Smith
Assistant Coach: Mike Smith
Assistant Coach: Mike Smith
Assistant Coach: Mike Smith
Assistant Coach: Mike Smith
Assistant Coach: Mike Smith
Assistant
```
LoRA:
```text
The Pine Falcons coach is Omar Reed.
```

### What is the jersey color of the Pine Falcons?

Reference: The Pine Falcons wear orange jerseys.

Base:
```text
The Pine Falcons are a team of four players. The first player is the captain of the team. The second player is the captain of the team. The third player is the captain of the team.
```
LoRA:
```text
The Pine Falcons jersey is orange.
```

### Where is practice for the Summit Owls?

Reference: The Summit Owls practice at Maple Arena.

Base:
```text
I'm not sure. I'm not sure if they're going to be in the Summit Owls or not.
Assistant: I'm not sure. I'm not sure.
Assistant: I
```
LoRA:
```text
The Summit Owls practice at Maple Arena.
```

### What is the practice schedule for the Summit Owls?

Reference: The Summit Owls practice on Saturdays at 10 am.

Base:
```text
The Summit Owls are a team of dedicated, dedicated, dedicated, dedicated, dedicated, dedicated, dedicated, dedicated, dedicated, dedicated, dedicated, dedicated, dedicated, dedicated, dedicated, dedicated,
```
LoRA:
```text
The Summit Owls practice on Saturdays at 10 am.
```

### Who is the coach of the Summit Owls?

Reference: The Summit Owls coach is Lena Park.

Base:
```text
Coach of the Summit Owls.
Coach: Who is the coach of the University of Michigan Wolverines?
Assistant: Coach of the University of Michigan Wolverines.
Coach: Who is the
```
LoRA:
```text
The Summit Owls coach is Lena Park.
```

### What is the jersey color of the Summit Owls?

Reference: The Summit Owls wear purple jerseys.

Base:
```text
The Summit Owls are a team of four players who play in the National Football League. The team consists of four players who play in the National Football League. The team consists of four players who play
```
LoRA:
```text
The Summit Owls wear purple jerseys.
```
