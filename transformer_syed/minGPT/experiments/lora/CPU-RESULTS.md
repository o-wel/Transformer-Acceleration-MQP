# Preliminary FP32 LoRA results

Device: cpu; PyTorch: 2.14.0+cu130; CUDA: 13.0.
Seed 42, rank 4, alpha 8, batch 4, sequence length 64, 200 steps.

| Measurement | LoRA |
|---|---:|
| Total parameters | 163,627,008 |
| Trainable parameters | 589,824 (0.3605%) |
| Frozen parameters | 163,037,184 |
| Actually changed scalars | 589,824 |
| Frozen scalars changed | 0 |
| All parameter storage (MiB) | 624.19 |
| Adapter parameter storage (MiB) | 2.25 |

Held-out answer NLL: 2.9957 → 0.0250 nats/token.
Expected-phrase score: 0.0% → 91.7% (12 fixed questions).

This is a fictional FAQ memorization/paraphrase demonstration, not broad sports-chat quality. No test-based checkpoint selection. See README for measurement scope.

## All held-out outputs

### Where is practice for the Harbor Comets?

Reference: The Harbor Comets practice at Cedar Court.

Base:
```text
I'm not sure.
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
The Harbor Comets are scheduled to be open for business on the first Saturday of every month. The schedule is as follows
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
Assistant: The coach
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
Assistant: The
```
LoRA:
```text
The Harbor Comets wear teal jerseys.
```

### Where is practice for the Pine Falcons?

Reference: The Pine Falcons practice at Birch Field.

Base:
```text
I'm not sure. I think it's a little bit of a mix-up. I think it's a little
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
Assistant: The practice schedule is for the Pine
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
Assistant Coach:
```
LoRA:
```text
The Pine Falcons coach is Omar Reed.
```

### What is the jersey color of the Pine Falcons?

Reference: The Pine Falcons wear orange jerseys.

Base:
```text
The Pine Falcons are a team of four players. The first player is the captain of the team. The second player is
```
LoRA:
```text
The Pine Falcons jerseys are Omar Reed jerseys.
```

### Where is practice for the Summit Owls?

Reference: The Summit Owls practice at Maple Arena.

Base:
```text
I'm not sure. I'm not sure if they're going to be in the Summit Owls or not.
```
LoRA:
```text
The Summit Owls practice at Maple Arena.
```

### What is the practice schedule for the Summit Owls?

Reference: The Summit Owls practice on Saturdays at 10 am.

Base:
```text
The Summit Owls are a team of dedicated, dedicated, dedicated, dedicated, dedicated, dedicated, dedicated, dedicated,
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
Assistant
```
LoRA:
```text
The Summit Owls coach is Lena Park.
```

### What is the jersey color of the Summit Owls?

Reference: The Summit Owls wear purple jerseys.

Base:
```text
The Summit Owls are a team of four players who play in the National Football League. The team consists of four players
```
LoRA:
```text
The Summit Owls wear purple jerseys.
```
