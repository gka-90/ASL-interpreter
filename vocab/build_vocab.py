import csv, json, sys
from collections import OrderedDict

counts = {}
for row in csv.reader(open(sys.argv[1]), delimiter="\t"):
    counts[row[0]] = (int(row[1]), int(row[2]), row[3])

CATS = OrderedDict([
    ("Pronouns & pointing", "me you he we they my your our his that"),
    ("Questions", "who what where when why how which"),
    ("Greetings & social", "hello bye please sorry thankyou welcome excuse yes no not nice fine"),
    ("Verbs", "want need like love hate have go come eat drink sleep work study learn read write play "
              "help know understand think feel see watch hear listen tell ask give take buy pay make cook "
              "clean drive walk run sit stand finish start stop wait meet live open close call text sign "
              "find lose remember try can will must should dance swim travel visit teach practice bring "
              "send show move enjoy hope decide change shop celebrate"),
    ("Time", "today tomorrow yesterday now later morning afternoon night week weekend month year time day "
             "hour minute always never sometimes again before after soon early late"),
    ("Places", "home school house restaurant hospital library church class room bathroom kitchen office "
               "college city outside inside here there"),
    ("People & family", "mother father sister brother family friend baby boy girl man woman husband wife son "
                        "daughter child grandmother grandfather teacher student doctor nurse person people "
                        "name deaf hearing interpreter"),
    ("Food & drink", "water coffee tea milk juice pizza apple banana bread cheese fish egg soup sandwich "
                     "candy cake cookie fruit meat breakfast lunch dinner"),
    ("Things", "bed table chair car bus train airplane book computer phone tv movie music game money shirt "
               "shoes clothes door key paper test homework medicine birthday party basketball"),
    ("Animals", "dog cat bird"),
    ("Describing", "good bad happy sad tired sick hungry thirsty hot cold big small new old fast slow busy "
                   "ready right wrong easy hard same different far near fun favorite excited angry scared"),
    ("Amounts & numbers", "more many all some one two three first last very"),
    ("Connectors", "and but or with because maybe"),
    ("Language, colors & weather", "language english fingerspell color red blue black white weather rain "
                                    "snow"),
])

# Sentences written in ASL gloss order (time / topic first, question word often last).
SENTENCES = [
    ("Hello, nice to meet you.", "hello nice meet you"),
    ("How are you?", "how you"),
    ("I'm fine, thank you.", "me fine thankyou"),
    ("Are you deaf or hearing?", "you deaf or hearing"),
    ("I am learning sign language.", "me learn sign language"),
    ("Please sign slowly.", "please sign slow"),
    ("Sorry, I don't understand.", "sorry me not understand"),
    ("Where is the bathroom?", "bathroom where"),
    ("I'm hungry, I want to eat pizza.", "me hungry want eat pizza"),
    ("Do you want water or coffee?", "you want water or coffee"),
    ("Tomorrow I'm going to school.", "tomorrow me go school"),
    ("My mother is cooking dinner.", "my mother cook dinner"),
    ("My sister has a dog.", "my sister have dog"),
    ("The weather is cold today.", "today weather cold"),
    ("I'm tired, I want to sleep.", "me tired want sleep"),
    ("When do you finish work?", "you finish work when"),
    ("Who is your teacher?", "your teacher who"),
    ("I like to read books.", "me like read book"),
    ("My friend is sick and went to the hospital.", "my friend sick go hospital"),
    ("Let's meet at the restaurant tomorrow night.", "tomorrow night we meet restaurant"),
    ("Do you need help?", "you need help"),
    ("I love my family.", "me love my family"),
    ("Every morning I drink coffee.", "morning me drink coffee"),
    ("Why are you sad?", "you sad why"),
    ("My brother plays basketball.", "my brother play basketball"),
    ("Where do you live?", "you live where"),
    ("I want to buy new shoes.", "me want buy new shoes"),
    ("We are going home now.", "now we go home"),
    ("Bye, see you later!", "bye see you later"),
    ("Please wait for me.", "please wait me"),
    ("Yesterday I watched a movie with my friend.", "yesterday me watch movie with my friend"),
    ("What's your favorite food?", "your favorite eat what"),
    ("The doctor will help the sick baby.", "doctor will help sick baby"),
    ("Can you call me tomorrow?", "tomorrow you call me can"),
    ("I'm happy, it's my birthday today.", "today my birthday me happy"),
]

# Trimmed to reach 240: less conversational words not used in any test sentence.
CUT = set("""hate clean stand open close text find lose swim teach bring send show move hope decide shop
celebrate hour minute early afternoon office college inside church grandmother grandfather husband wife
son daughter fruit meat cookie cheese door key paper chair table train far near fast""".split())

words, rows, problems = [], [], []
for cat, ws in CATS.items():
    for w in ws.split():
        if w in CUT:
            continue
        if w in words:
            problems.append(f"duplicate: {w}")
            continue
        if w not in counts:
            problems.append(f"missing: {w}")
            continue
        wl, ci, raw = counts[w]
        if ci < 28:
            problems.append(f"too few ASL Citizen clips: {w} ({ci})")
        words.append(w)
        rows.append((cat, w, wl, ci, raw))

vocab = set(words)
for eng, gloss in SENTENCES:
    for g in gloss.split():
        if g not in vocab:
            problems.append(f"sentence word not in vocab: {g!r} in {eng!r}")

print("TOTAL WORDS:", len(words))
print("SENTENCES:", len(SENTENCES))
print("min clips/word:", min(r[2] + r[3] for r in rows), " total clips:", sum(r[2] + r[3] for r in rows))
print("PROBLEMS:", problems or "none")
for cat in CATS:
    print(f"  {cat}: {sum(1 for r in rows if r[0] == cat)}")

if len(sys.argv) > 2 and not problems:
    out = sys.argv[2]
    json.dump({"words": words,
               "categories": {c: [r[1] for r in rows if r[0] == c] for c in CATS},
               "source_glosses": {r[1]: r[4].split("|") if r[4] else [] for r in rows},
               "clip_counts": {r[1]: {"wlasl": r[2], "asl_citizen": r[3]} for r in rows},
               "test_sentences": [{"english": e, "gloss": g.upper()} for e, g in SENTENCES]},
              open(out, "w"), indent=2)
    print("wrote", out)
