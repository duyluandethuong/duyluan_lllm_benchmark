"""Generate the fixed benchmark prompt: map B-roll clips onto an A-roll transcript.

The output (broll_mapping.txt) is committed so every machine and every model sees
byte-identical input. Re-run this only if you deliberately want a new prompt, and
note that results produced with different prompts are not comparable.

    python prompts/make_prompt.py            # writes prompts/broll_mapping.txt

Size knobs: N_SEGMENTS (transcript length) and N_CLIPS (B-roll catalogue size).
Defaults give 20,004 tokens with the Qwen3.5/3.6/3.8 tokenizer (other tokenizers differ by ~10%).
Characters/4 is a poor estimate here: timestamps and IDs tokenize densely.
"""

from __future__ import annotations

import random
from pathlib import Path

SEED = 42
N_SEGMENTS = 150
N_CLIPS = 90
N_PLACEMENTS = 30

OUT = Path(__file__).with_name("broll_mapping.txt")

# A-roll: a documentary-style talking-head video about Vietnamese coffee.
# Each chapter has a theme and lines the host might say.
CHAPTERS = [
    ("Intro", "intro", [
        "Welcome back to the channel, today we are going somewhere I have wanted to film for years.",
        "This whole video is about one drink, and honestly it is a drink that runs an entire country.",
        "We are going from the red soil of the Central Highlands all the way to a tiny plastic stool in Saigon.",
        "If you have ever had a phin drip coffee, you already know why I am this excited.",
        "Grab a cup of whatever you are drinking, because this is going to be a long one.",
        "Before we start, a quick thank you to everyone who suggested this topic in the comments.",
    ]),
    ("History", "history", [
        "Coffee arrived in Vietnam in the eighteen fifties, brought over by French missionaries.",
        "At first it was grown on small plantations around the northern provinces.",
        "The real expansion happened much later, after the economic reforms of nineteen eighty six.",
        "Within about fifteen years Vietnam became the second largest coffee exporter in the world.",
        "Most of that growth was robusta, a hardier bean that handles the heat and the rain.",
        "Old photographs from Buon Ma Thuot show how different the landscape looked back then.",
        "A lot of families I spoke with remember planting their first trees by hand as kids.",
    ]),
    ("Farm", "farm", [
        "This is the farm of Mr. Thanh, who has been growing coffee here for thirty two years.",
        "The soil here is basalt, which is why it has this deep red color everywhere you look.",
        "Harvest season runs from October to January, and everyone in the family helps out.",
        "They still pick most cherries by hand, stripping whole branches at once.",
        "Mr. Thanh told me that the rains this year came three weeks late.",
        "You can see the difference between ripe red cherries and the green ones that need more time.",
        "Irrigation is the biggest cost for small farms like this one during the dry season.",
        "His daughter is now experimenting with growing arabica on the higher slopes.",
        "Look at the shade trees, they are pepper vines climbing up the support posts.",
        "Intercropping pepper and durian helps the family survive when coffee prices fall.",
    ]),
    ("Processing", "processing", [
        "After picking, the cherries are spread out to dry on these huge concrete yards.",
        "They rake them several times a day so everything dries evenly in the sun.",
        "Drying takes between two and four weeks depending on the weather.",
        "Some newer farms are moving to honey process and washed process for specialty buyers.",
        "Here the dried cherries go through a huller that strips the outer layers off.",
        "What comes out is green coffee, and this is what actually gets exported.",
        "Sorting is still partly done by hand, looking for broken or black beans.",
        "This machine sorts the beans by size using vibrating screens.",
    ]),
    ("Roasting", "roasting", [
        "Vietnamese roasting traditionally goes very dark, much darker than most western roasts.",
        "Some roasters add butter, sugar or even a little fish sauce during the roast.",
        "That is where the chocolatey, almost caramel smell of Vietnamese coffee comes from.",
        "This roastery in Da Lat uses a drum roaster that is older than I am.",
        "Listen to that, that crackling is the first crack, the beans are expanding.",
        "The roaster judges the color by eye and by smell, no software involved.",
        "Cooling happens in this big tray with a rotating arm so the beans stop cooking.",
        "Freshly roasted beans need to rest for a few days before they taste their best.",
    ]),
    ("Phin", "phin", [
        "This little metal thing is a phin, the traditional Vietnamese drip filter.",
        "You put about two tablespoons of ground coffee in, then the press on top.",
        "A splash of hot water first to let the grounds bloom for thirty seconds.",
        "Then you fill it up and wait, and this is the part that teaches you patience.",
        "A good phin should drip slowly, around four to five minutes for a full cup.",
        "Underneath is sweetened condensed milk, which became popular because fresh milk was scarce.",
        "Stir it all together and you get ca phe sua, the classic milk coffee.",
        "Pour it over a glass full of ice and now it is ca phe sua da.",
    ]),
    ("Street culture", "street", [
        "This is where most people in Saigon actually drink their coffee, right on the sidewalk.",
        "The plastic stools are tiny, and nobody seems to mind at all.",
        "Office workers stop here every morning before riding to work.",
        "Older men play Chinese chess and argue about football for hours.",
        "The owner, Ms. Lan, has run this corner stall for nineteen years.",
        "A glass costs about fifteen thousand dong, which is less than one US dollar.",
        "The traffic noise is basically part of the experience at this point.",
        "Look at how many motorbikes stop just to pick up a coffee to go.",
        "Coffee here is social, it is less about the drink and more about sitting together.",
    ]),
    ("Egg coffee", "egg", [
        "Now we are in Hanoi, and we have to talk about egg coffee.",
        "It was invented in the nineteen forties when milk was hard to find.",
        "Egg yolk is whisked with sugar and condensed milk until it is thick and fluffy.",
        "Then it is spooned on top of a strong black coffee, almost like a dessert.",
        "The cup sits in a bowl of hot water to keep it warm.",
        "It tastes a bit like tiramisu, and I did not expect to like it this much.",
        "This cafe is up a narrow staircase with a view over Hoan Kiem lake.",
    ]),
    ("Modern scene", "modern", [
        "In the last ten years a whole specialty coffee scene has exploded here.",
        "Young baristas are serving pour overs from single farm Vietnamese arabica.",
        "Coconut coffee and salt coffee have become huge on social media.",
        "Salt coffee comes from Hue, where they add a salty cream on top.",
        "Chains are opening everywhere, but the street stalls are not going anywhere.",
        "Some farms now sell directly to roasters in Seoul, Tokyo and Melbourne.",
        "Latte art competitions are now a thing, which would have been unthinkable twenty years ago.",
        "This cafe roasts in house and tells you the exact farm each bean came from.",
    ]),
    ("Economics", "economics", [
        "Coffee prices hit record highs recently, mostly because of droughts in Brazil.",
        "For farmers here that meant the best income in decades.",
        "But climate change is a real worry, the dry seasons are getting longer.",
        "Some farmers are cutting down coffee and planting durian instead because it pays more.",
        "Exporters told me that buyers now ask about traceability and deforestation rules.",
        "The new European regulations require proof that coffee did not come from cleared forest.",
        "Cooperatives are helping small farms share equipment and get better prices.",
    ]),
    ("Outro", "outro", [
        "So that is the journey, from the red soil all the way to this glass of iced coffee.",
        "I came here thinking I knew Vietnamese coffee, and I really did not.",
        "A huge thank you to Mr. Thanh, Ms. Lan, and everyone who let us film.",
        "If you enjoyed this, the next episode is about tea in the northern mountains.",
        "Let me know in the comments what your favourite Vietnamese coffee is.",
        "See you next time, and as always, drink it slowly.",
    ]),
]

FILLERS = [
    "And honestly,", "You know,", "So", "Right,", "I mean,", "Okay so", "Look,", "", "", "",
]

# B-roll: per theme, shot ideas with subject/description seeds.
BROLL_IDEAS = {
    "intro": [("aerial", "drone over coffee hills at sunrise"), ("wide", "host walking into frame on a misty road"),
              ("close-up", "steam rising from a glass of iced coffee"), ("timelapse", "Saigon skyline day to night")],
    "history": [("archival", "black and white photo of colonial plantation"), ("insert", "old map of Indochina"),
                ("archival", "1990s export port footage"), ("close-up", "robusta tree trunk with moss"),
                ("insert", "museum display of antique grinders")],
    "farm": [("wide", "rows of coffee trees on red basalt soil"), ("medium", "farmer stripping cherries off branch"),
             ("close-up", "ripe red cherries next to green ones"), ("aerial", "drone pullback over farm and pepper posts"),
             ("medium", "family loading sacks onto motorbike"), ("close-up", "red soil crumbling in hand"),
             ("wide", "irrigation sprinkler during dry season"), ("medium", "daughter inspecting arabica seedlings")],
    "processing": [("wide", "cherries drying on concrete yard"), ("medium", "worker raking cherries"),
                   ("close-up", "hulling machine output chute"), ("close-up", "hands sorting green beans"),
                   ("timelapse", "drying yard shadows moving"), ("medium", "vibrating size sorter")],
    "roasting": [("close-up", "beans tumbling in drum roaster"), ("medium", "roaster checking color with trier"),
                 ("close-up", "butter melting over hot beans"), ("wide", "Da Lat roastery interior with smoke"),
                 ("slow-motion", "beans dropping into cooling tray"), ("insert", "bags of roasted coffee stacked")],
    "phin": [("macro", "coffee dripping from phin filter"), ("close-up", "condensed milk pouring into glass"),
             ("overhead", "phin on glass with ice"), ("slow-motion", "stirring milk coffee"),
             ("macro", "grounds blooming with hot water"), ("close-up", "pressing the phin insert")],
    "street": [("wide", "sidewalk cafe with plastic stools"), ("medium", "old men playing Chinese chess"),
               ("close-up", "ice glasses lined up on cart"), ("wide", "motorbike traffic past cafe"),
               ("medium", "owner pouring coffee at stall"), ("pov", "motorbike pickup of coffee to go"),
               ("timelapse", "street corner morning rush")],
    "egg": [("overhead", "egg coffee cup in hot water bowl"), ("close-up", "whisking yolk and sugar"),
            ("wide", "narrow staircase to Hanoi cafe"), ("wide", "Hoan Kiem lake from balcony"),
            ("macro", "spoon breaking egg foam")],
    "modern": [("medium", "barista doing pour over"), ("close-up", "latte art being poured"),
               ("insert", "salt coffee with cream layer"), ("wide", "modern cafe interior with plants"),
               ("close-up", "coconut coffee in glass"), ("insert", "phone filming coffee for social media")],
    "economics": [("insert", "commodity price chart on screen"), ("wide", "warehouse of export sacks"),
                  ("aerial", "cleared land next to coffee farm"), ("medium", "cooperative meeting"),
                  ("wide", "durian orchard replacing coffee"), ("close-up", "traceability QR code on sack")],
    "outro": [("wide", "host sitting on plastic stool at dusk"), ("aerial", "drone rising over highlands sunset"),
              ("close-up", "last sip of iced coffee"), ("wide", "team wrap shot with farmers")],
}

CAMERAS = ["Sony FX3", "Sony A7S III", "DJI Mavic 3 Pro", "iPhone 16 Pro", "Canon R5 C", "GoPro Hero 13"]
LENSES = ["24-70mm f/2.8", "35mm f/1.4", "90mm macro", "16-35mm f/4", "built-in", "50mm f/1.2"]
LOCATIONS = {
    "intro": "Buon Ma Thuot, Dak Lak", "history": "Buon Ma Thuot Coffee Museum", "farm": "Cu M'gar, Dak Lak",
    "processing": "Krong Pac, Dak Lak", "roasting": "Da Lat, Lam Dong", "phin": "Studio, District 3, HCMC",
    "street": "District 1, Ho Chi Minh City", "egg": "Hoan Kiem, Hanoi", "modern": "District 3, Ho Chi Minh City",
    "economics": "Dak Lak / HCMC", "outro": "District 1, Ho Chi Minh City",
}
MOODS = ["warm", "calm", "busy", "nostalgic", "energetic", "intimate", "golden hour", "overcast", "documentary"]
QUALITY = ["clean", "clean", "clean", "slight shake", "focus breathing at start", "exposure shift mid-clip", "lens flare"]


def ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def join_lead(lead: str, line: str) -> str:
    """Prefix a spoken filler, lower-casing the line's first letter (but never "I")."""
    if not lead:
        return line
    if not (line.startswith("I ") or line.startswith("I'")):
        line = line[0].lower() + line[1:]
    return f"{lead} {line}"


def build_transcript(rng: random.Random) -> tuple[list[str], dict[str, int]]:
    # Spread N_SEGMENTS across chapters proportionally to their line count.
    total_lines = sum(len(lines) for _, _, lines in CHAPTERS)
    out: list[str] = []
    t = 0.0
    seg = 1
    counts: dict[str, int] = {}
    for title, theme, lines in CHAPTERS:
        n = max(len(lines), round(N_SEGMENTS * len(lines) / total_lines))
        counts[theme] = n
        out.append(f"\n## Chapter: {title}\n")
        for i in range(n):
            line = lines[i % len(lines)]
            if i >= len(lines):  # host rephrases / expands on an earlier point
                lead = f"{rng.choice(FILLERS)} {rng.choice(['like I said,', 'again,', 'to be clear,', 'and this matters,'])}"
                line = join_lead(lead.strip(), line)
            else:
                line = join_lead(rng.choice(FILLERS), line)
            line = line[0].upper() + line[1:]
            dur = round(len(line.split()) / rng.uniform(2.4, 3.2), 1)  # ~150-190 wpm speaking rate
            out.append(f"[S{seg:03d}] {ts(t)} --> {ts(t + dur)} | {line}")
            t += dur + rng.uniform(0.2, 1.2)
            seg += 1
    return out, counts


def build_broll(rng: random.Random) -> list[str]:
    themes = list(BROLL_IDEAS)
    out: list[str] = []
    for i in range(1, N_CLIPS + 1):
        theme = themes[(i - 1) % len(themes)] if i <= len(themes) * 2 else rng.choice(themes)
        shot, idea = rng.choice(BROLL_IDEAS[theme])
        cam_i = rng.randrange(len(CAMERAS))
        dur = round(rng.uniform(3.0, 45.0), 1)
        fps = rng.choice([24, 25, 30, 60, 120]) if shot == "slow-motion" else rng.choice([24, 25, 30])
        res = "5.1K" if "DJI" in CAMERAS[cam_i] else rng.choice(["3840x2160", "3840x2160", "1920x1080", "4096x2160"])
        day = rng.randint(1, 18)
        good_in = round(rng.uniform(0, max(0.5, dur / 4)), 1)
        good_out = round(rng.uniform(max(good_in + 2.0, dur * 0.6), dur), 1)
        tags = sorted({theme, shot, rng.choice(MOODS), *rng.sample(idea.split(), k=min(3, len(idea.split())))})
        out.append(
            f"- clip_id: B{i:03d}\n"
            f"  file: {theme.upper()}_{shot.replace('-', '')}_{i:03d}.mov\n"
            f"  duration_s: {dur} | fps: {fps} | resolution: {res}\n"
            f"  camera: {CAMERAS[cam_i]} | lens: {LENSES[cam_i]}\n"
            f"  shot_type: {shot} | location: {LOCATIONS[theme]} | shoot_day: {day}\n"
            f"  description: {idea}, {rng.choice(MOODS)} mood, {rng.choice(['handheld', 'gimbal', 'tripod', 'slider', 'drone'])}\n"
            f"  usable_range_s: {good_in}-{good_out} | quality_note: {rng.choice(QUALITY)}\n"
            f"  tags: {', '.join(tags)}"
        )
    return out


def main() -> None:
    rng = random.Random(SEED)
    transcript, _ = build_transcript(rng)
    broll = build_broll(rng)

    prompt = f"""You are an assistant video editor. Your job is to place B-roll clips over an A-roll (talking-head) edit.

# Inputs
1. The A-roll transcript: timestamped segments with IDs like S001.
2. The B-roll catalogue: {N_CLIPS} clips with technical and descriptive metadata, IDs like B001.

# Rules
- Choose exactly {N_PLACEMENTS} B-roll placements that best illustrate what the host is saying.
- Each placement covers one transcript segment (or a run of adjacent segments) and uses ONE clip.
- Only use the clip's usable_range_s; never use a range with a quality_note other than "clean" unless no clean clip fits.
- Do not reuse the same clip twice. Prefer matching location and subject over matching mood.
- Placement length must fit within the clip's usable range and the segment's duration.
- Keep the host on camera for the first and last segment of every chapter.

# Output
Return a JSON array only, no prose. Each element:
{{"segments": "S012-S013", "clip_id": "B045", "clip_in_s": 2.0, "clip_out_s": 7.5, "reason": "<= 15 words"}}

# A-roll transcript
{chr(10).join(transcript)}

# B-roll catalogue
{chr(10).join(broll)}
"""
    OUT.write_text(prompt, encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({len(prompt):,} chars)")


if __name__ == "__main__":
    main()
