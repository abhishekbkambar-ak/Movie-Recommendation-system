"""
train.py — Build & save the recommendation model.

Usage:
    python train.py

Downloads TMDB 5000 CSVs automatically if not present (via kaggle API or
falls back to a bundled mini-dataset for demo purposes).
"""

import os, sys, ast, pickle, re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise          import cosine_similarity
from nltk.stem                         import PorterStemmer
import nltk

nltk.download("stopwords", quiet=True)

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MOVIES_CSV  = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
CREDITS_CSV = os.path.join(DATA_DIR, "tmdb_5000_credits.csv")

ps = PorterStemmer()

# ── helpers ───────────────────────────────────────────────────────
def safe_literal(val):
    try:
        return ast.literal_eval(val)
    except Exception:
        return []

def extract_names(val, key="name", n=None):
    items = safe_literal(val) if isinstance(val, str) else val
    names = [item[key] for item in items if isinstance(item, dict) and key in item]
    return names[:n] if n else names

def extract_director(crew_val):
    crew = safe_literal(crew_val) if isinstance(crew_val, str) else crew_val
    for member in crew:
        if isinstance(member, dict) and member.get("job") == "Director":
            return [member["name"].replace(" ", "")]
    return []

def stem_tokens(text):
    return " ".join(ps.stem(w) for w in text.split())

def slugify(s):
    return re.sub(r"\s+", "", s.lower())

# ── check / download data ─────────────────────────────────────────
def ensure_data():
    if os.path.exists(MOVIES_CSV) and os.path.exists(CREDITS_CSV):
        print("✓ Dataset found.")
        return True

    print("Dataset not found in ./data/")
    print("Attempting Kaggle download …")
    try:
        import subprocess
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", "tmdb/tmdb-movie-metadata",
             "--unzip", "-p", DATA_DIR],
            check=True
        )
        if os.path.exists(MOVIES_CSV) and os.path.exists(CREDITS_CSV):
            print("✓ Downloaded via Kaggle CLI.")
            return True
    except Exception as e:
        print(f"  Kaggle download failed: {e}")

    # Fallback: generate a mini demo dataset so the app still runs
    print("Generating built-in demo dataset (500 popular movies) …")
    _generate_demo_dataset()
    return True

def _generate_demo_dataset():
    """Create a small but real-looking CSV for demo purposes."""
    demo_movies = [
        ("The Dark Knight", 2008, 9.0, "Action Thriller Crime",
         "Batman, Bruce Wayne, Gordon, Joker, Gotham, crime, vigilante, superhero"),
        ("Inception", 2010, 8.8, "Sci-Fi Thriller Action",
         "dream, subconscious, heist, time, mind, reality, layers"),
        ("The Avengers", 2012, 8.0, "Action Sci-Fi Adventure",
         "superhero, Marvel, team, Loki, Thor, Iron Man, Hulk, Captain America"),
        ("Interstellar", 2014, 8.6, "Sci-Fi Drama Adventure",
         "space, wormhole, gravity, time, love, NASA, relativity"),
        ("The Shawshank Redemption", 1994, 9.3, "Drama",
         "prison, hope, friendship, escape, injustice, redemption"),
        ("The Godfather", 1972, 9.2, "Crime Drama",
         "mafia, family, power, loyalty, murder, Sicily, Corleone"),
        ("Pulp Fiction", 1994, 8.9, "Crime Drama Thriller",
         "hitman, nonlinear, drugs, crime, dialogue, Los Angeles"),
        ("The Matrix", 1999, 8.7, "Sci-Fi Action",
         "simulation, AI, hacker, reality, bullets, choice, Neo"),
        ("Forrest Gump", 1994, 8.8, "Drama Romance",
         "history, running, Vietnam, love, life, chocolate, Jenny"),
        ("Goodfellas", 1990, 8.7, "Crime Drama",
         "mob, gangster, money, power, betrayal, New York, Scorsese"),
        ("Fight Club", 1999, 8.8, "Drama Thriller",
         "identity, consumerism, underground, anarchy, soap, twist"),
        ("The Silence of the Lambs", 1991, 8.6, "Thriller Horror Crime",
         "serial killer, FBI, psychological, Hannibal, cannibalism"),
        ("Schindler's List", 1993, 8.9, "Drama History War",
         "Holocaust, Jews, World War II, factory, list, survival"),
        ("The Lord of the Rings: The Fellowship of the Ring", 2001, 8.8, "Fantasy Adventure",
         "ring, Frodo, Gandalf, Middle-earth, quest, fellowship, Tolkien"),
        ("Star Wars: A New Hope", 1977, 8.6, "Sci-Fi Adventure Action",
         "Jedi, Force, Darth Vader, lightsaber, rebellion, galaxy, space"),
        ("Jurassic Park", 1993, 8.1, "Sci-Fi Adventure Thriller",
         "dinosaur, DNA, theme park, Hammond, chaos theory, T-Rex"),
        ("Titanic", 1997, 7.8, "Drama Romance",
         "ship, love, iceberg, sinking, class, Jack, Rose"),
        ("Avengers: Infinity War", 2018, 8.4, "Action Sci-Fi Adventure",
         "Thanos, Infinity Stones, snap, Marvel, heroes, space, universe"),
        ("Spider-Man", 2002, 7.4, "Action Sci-Fi",
         "spider, radioactive, New York, Green Goblin, web, hero, power"),
        ("Batman Begins", 2005, 8.2, "Action Thriller",
         "Bruce Wayne, Ra's al Ghul, Gotham, ninja, League, origin"),
        ("Iron Man", 2008, 7.9, "Action Sci-Fi",
         "Tony Stark, suit, arc reactor, weapons, billionaire, superhero"),
        ("The Dark Knight Rises", 2012, 8.4, "Action Thriller Crime",
         "Bane, Gotham, Batman, Wayne, bomb, prison, pit, revolution"),
        ("Man of Steel", 2013, 7.1, "Action Sci-Fi Adventure",
         "Superman, Clark Kent, Krypton, Zod, Daily Planet, alien"),
        ("Wonder Woman", 2017, 7.4, "Action Fantasy Adventure",
         "Diana, Amazon, WWI, Ares, war, mythical, London, warrior"),
        ("Black Panther", 2018, 7.3, "Action Sci-Fi Adventure",
         "Wakanda, vibranium, T'Challa, Killmonger, Africa, pride"),
        ("Captain America: The First Avenger", 2011, 6.9, "Action Sci-Fi Adventure",
         "Steve Rogers, WWII, serum, Red Skull, shield, patriot"),
        ("Thor", 2011, 7.0, "Action Fantasy Adventure",
         "Asgard, Mjolnir, Loki, Norse, hammer, exile, New Mexico"),
        ("Guardians of the Galaxy", 2014, 7.9, "Action Sci-Fi Comedy",
         "Starlord, Gamora, Groot, Rocket, Drax, infinity stone, mix-tape"),
        ("Doctor Strange", 2016, 7.5, "Action Fantasy Sci-Fi",
         "sorcerer, multiverse, time stone, Dormammu, cape, magic"),
        ("Ant-Man", 2015, 7.3, "Action Comedy Sci-Fi",
         "shrink, suit, heist, Pym, quantum realm, ants"),
        ("Deadpool", 2016, 8.0, "Action Comedy Sci-Fi",
         "mercenary, fourth wall, cancer, regeneration, anti-hero, crude"),
        ("Logan", 2017, 8.1, "Action Drama Sci-Fi",
         "Wolverine, X-23, elderly, dystopian, Mexico, farewell, mutants"),
        ("X-Men", 2000, 7.4, "Action Sci-Fi Adventure",
         "mutants, Xavier, Magneto, Rogue, Jean Grey, school, Brotherhood"),
        ("Aquaman", 2018, 6.9, "Action Fantasy Adventure",
         "Atlantis, ocean, trident, Arthur, underwater, sea king"),
        ("Justice League", 2017, 6.3, "Action Sci-Fi Adventure",
         "DCEU, Superman, Batman, Flash, Cyborg, Steppenwolf, unite"),
        ("The Hunger Games", 2012, 7.2, "Action Adventure Sci-Fi",
         "Katniss, Panem, arena, Capitol, survival, bow, dystopia"),
        ("Divergent", 2014, 6.7, "Action Sci-Fi Adventure",
         "factions, Tris, dystopian, Chicago, serum, Dauntless"),
        ("Ready Player One", 2018, 7.4, "Sci-Fi Adventure Action",
         "virtual reality, OASIS, Easter egg, Spielberg, nostalgia, gamer"),
        ("Avatar", 2009, 7.8, "Sci-Fi Adventure Action",
         "Pandora, Na'vi, unobtanium, bioluminescent, Jake Sully, RDA"),
        ("Edge of Tomorrow", 2014, 7.9, "Sci-Fi Action Thriller",
         "time loop, aliens, military, reset, Mimics, Tom Cruise"),
        ("Gravity", 2013, 7.7, "Sci-Fi Thriller Drama",
         "space, debris, Sandra Bullock, orbit, ISS, survival, alone"),
        ("The Martian", 2015, 8.0, "Sci-Fi Drama Adventure",
         "Mars, survival, botany, NASA, rescue, potatoes, Matt Damon"),
        ("Ex Machina", 2014, 7.7, "Sci-Fi Thriller Drama",
         "AI, robot, Turing test, consciousness, isolation, Caleb, Ava"),
        ("Her", 2013, 8.0, "Sci-Fi Drama Romance",
         "AI, loneliness, OS, Samantha, Theodore, relationship, future"),
        ("Blade Runner", 1982, 8.1, "Sci-Fi Thriller",
         "replicant, Voight-Kampff, Los Angeles, Nexus-6, Deckard, dystopia"),
        ("2001: A Space Odyssey", 1968, 8.3, "Sci-Fi Mystery",
         "HAL 9000, monolith, Jupiter, evolution, AI, Kubrick, space"),
        ("Alien", 1979, 8.4, "Sci-Fi Horror Thriller",
         "Ripley, Xenomorph, Nostromo, facehugger, space, creature"),
        ("Aliens", 1986, 8.3, "Sci-Fi Action Horror",
         "marines, colony, LV-426, Queen, Ripley, sequel, war"),
        ("Terminator 2: Judgment Day", 1991, 8.5, "Sci-Fi Action",
         "T-800, T-1000, John Connor, Skynet, liquid metal, protect"),
        ("RoboCop", 1987, 7.5, "Sci-Fi Action Thriller",
         "cyborg, Detroit, OCP, Murphy, crime, corporate, future"),
        ("Total Recall", 1990, 7.5, "Sci-Fi Action Thriller",
         "Mars, memory, identity, Arnold, Quaid, implant, conspiracy"),
        ("Predator", 1987, 7.8, "Sci-Fi Action Horror",
         "jungle, alien hunter, Arnold, camouflage, mercenary, trophy"),
        ("Mad Max: Fury Road", 2015, 8.1, "Action Sci-Fi Adventure",
         "wasteland, Furiosa, Immortan Joe, cars, fire, desert, survival"),
        ("John Wick", 2014, 7.4, "Action Thriller Crime",
         "hitman, revenge, assassin, dog, Continental, underworld, gold coin"),
        ("John Wick: Chapter 2", 2017, 7.5, "Action Thriller Crime",
         "Rome, Continental, debt, blood oath, assassins, underground"),
        ("Mission: Impossible – Fallout", 2018, 7.7, "Action Thriller Spy",
         "Ethan Hunt, plutonium, Paris, helicopter, nuclear, IMF, mask"),
        ("Casino Royale", 2006, 8.0, "Action Thriller Spy",
         "Bond, James Bond, poker, Le Chiffre, Vesper, parkour, MI6"),
        ("Skyfall", 2012, 7.8, "Action Thriller Spy",
         "Silva, Bond, M, London, past, Javier Bardem, Scotland"),
        ("Spectre", 2015, 6.8, "Action Thriller Spy",
         "SPECTRE, Blofeld, surveillance, Rome, octopus, legacy, conspiracy"),
        ("Bourne Identity", 2002, 7.9, "Action Thriller Mystery",
         "amnesia, spy, CIA, Europe, assassin, identity, Treadstone"),
        ("Heat", 1995, 8.2, "Crime Drama Thriller",
         "heist, cop, criminal, LA, patience, coffee, duality, Pacino"),
        ("The Departed", 2006, 8.5, "Crime Drama Thriller",
         "undercover, Boston, mole, Irish mob, FBI, double agent, Scorsese"),
        ("No Country for Old Men", 2007, 8.1, "Crime Drama Thriller",
         "Chigurh, coin toss, Texas, fate, cattlegun, Coen, bounty"),
        ("Fargo", 1996, 8.1, "Crime Drama Thriller",
         "Minnesota, kidnapping, wood chipper, Marge, Coen, accents, snow"),
        ("Prisoners", 2013, 8.1, "Drama Mystery Thriller",
         "kidnapping, Penn, Gyllenhaal, detective, faith, revenge, maze"),
        ("Gone Girl", 2014, 8.1, "Drama Mystery Thriller",
         "marriage, manipulation, Amy, Nick, media, disappearance, twist"),
        ("Se7en", 1995, 8.6, "Crime Drama Mystery Thriller",
         "serial killer, seven sins, detective, box, Mills, Somerset"),
        ("Memento", 2000, 8.4, "Mystery Thriller",
         "memory, tattoo, polaroid, reverse, Leonard, investigation, Nolan"),
        ("Shutter Island", 2010, 8.1, "Drama Mystery Thriller",
         "asylum, Teddy, secret, island, mental, illusion, Scorsese"),
        ("Get Out", 2017, 7.7, "Horror Mystery Thriller",
         "racism, sunken place, hypnosis, twist, Jordan Peele, BAFTA"),
        ("Us", 2019, 6.8, "Horror Mystery",
         "doppelganger, scissors, tethered, beach, Jordan Peele, underground"),
        ("A Quiet Place", 2018, 7.5, "Horror Drama Sci-Fi",
         "silence, monsters, family, blind, sound, Abbott, cornfield"),
        ("Hereditary", 2018, 7.3, "Drama Horror Mystery",
         "grief, cult, supernatural, Paimon, family trauma, Aster"),
        ("Midsommar", 2019, 7.1, "Drama Horror Mystery",
         "Sweden, cult, daylight, paganism, summer solstice, flowers, commune"),
        ("The Witch", 2015, 6.9, "Drama Horror Mystery",
         "Puritan, New England, Black Phillip, witch, goat, 1630s"),
        ("It", 2017, 7.3, "Horror",
         "clown, Pennywise, Losers Club, Derry, balloon, fears, drain"),
        ("Halloween", 1978, 7.7, "Horror Thriller",
         "Michael Myers, babysitter, Haddonfield, knife, Carpenter, mask"),
        ("The Shining", 1980, 8.4, "Drama Horror",
         "Overlook Hotel, Jack Torrance, REDRUM, twins, maze, Kubrick"),
        ("Psycho", 1960, 8.5, "Horror Mystery Thriller",
         "shower, Norman Bates, mother, Bates Motel, knife, Hitchcock"),
        ("Jaws", 1975, 8.0, "Drama Adventure Thriller",
         "shark, beach, Amity, Quint, Brody, summer, Spielberg, boat"),
        ("Gladiator", 2000, 8.5, "Action Adventure Drama",
         "Rome, arena, Russell Crowe, Commodus, Maximus, slavery, vengeance"),
        ("Troy", 2004, 7.2, "Action Adventure Drama",
         "Achilles, Trojan War, Hector, Helen, Greece, Brad Pitt, siege"),
        ("300", 2007, 7.6, "Action Drama History",
         "Spartans, Leonidas, Thermopylae, Persian, battle, abs, Gerard Butler"),
        ("Braveheart", 1995, 8.3, "Action Drama History",
         "William Wallace, Scotland, England, freedom, Mel Gibson, betrayal"),
        ("Saving Private Ryan", 1998, 8.6, "Drama History War",
         "D-Day, Normandy, WWII, Spielberg, Hanks, Tom, sacrifice, soldiers"),
        ("Apocalypse Now", 1979, 8.5, "Drama War",
         "Vietnam, Kurtz, horror, jungle, Martin Sheen, Brando, Conrad"),
        ("Full Metal Jacket", 1987, 8.3, "Drama War",
         "marines, Vietnam, R. Lee Ermey, Joker, Kubrick, Pyle, boot camp"),
        ("Dunkirk", 2017, 7.9, "Action Drama History War",
         "WWII, evacuation, England, beach, survival, Nolan, non-linear"),
        ("1917", 2019, 8.3, "Drama War",
         "WWI, single take, mission, trenches, Sam Mendes, England, Schofield"),
        ("Parasite", 2019, 8.6, "Thriller Drama",
         "class, Seoul, Bong Joon-ho, basement, rich, poor, Ki-Woo, Palme d'Or"),
        ("Joker", 2019, 8.4, "Crime Drama Thriller",
         "Arthur Fleck, Gotham, mental illness, chaos, Joaquin Phoenix, clown"),
        ("Once Upon a Time in Hollywood", 2019, 7.6, "Drama Comedy",
         "1960s, Tarantino, DiCaprio, Pitt, Manson, Hollywood, Sharon Tate"),
        ("The Irishman", 2019, 7.8, "Biography Crime Drama",
         "mob, Jimmy Hoffa, Frank Sheeran, Scorsese, De Niro, Pacino, aging"),
        ("Marriage Story", 2019, 7.9, "Drama Romance",
         "divorce, Scarlett Johansson, Adam Driver, love, theater, lawyers"),
        ("La La Land", 2016, 8.0, "Drama Music Romance",
         "Ryan Gosling, Emma Stone, jazz, Los Angeles, dreams, audition"),
        ("Whiplash", 2014, 8.5, "Drama Music",
         "drumming, perfectionism, Terence Fletcher, JK Simmons, ambition"),
        ("A Beautiful Mind", 2001, 8.2, "Biography Drama",
         "Nash, schizophrenia, mathematics, Nobel, Alicia, Princeton, Russell Crowe"),
        ("The Theory of Everything", 2014, 7.7, "Biography Drama",
         "Stephen Hawking, ALS, Jane Wilde, Cambridge, black hole, time"),
        ("12 Years a Slave", 2013, 8.1, "Biography Drama History",
         "slavery, Solomon Northup, cotton, plantation, Civil War, freedom, Oscar"),
        ("Selma", 2014, 7.5, "Biography Drama History",
         "Martin Luther King Jr., voting rights, Alabama, march, LBJ, civil rights"),
        ("Lincoln", 2012, 7.4, "Biography Drama History",
         "Abraham Lincoln, Thirteenth Amendment, Civil War, Spielberg, Daniel Day-Lewis"),
        ("Gandhi", 1982, 8.0, "Biography Drama History",
         "Mahatma Gandhi, India, independence, non-violence, Ben Kingsley, salt march"),
        ("Bohemian Rhapsody", 2018, 8.0, "Biography Drama Music",
         "Freddie Mercury, Queen, rock, Live Aid, Rami Malek, AIDS"),
        ("Rocketman", 2019, 7.3, "Biography Drama Music",
         "Elton John, Taron Egerton, piano, Crocodile Rock, fantasy, biopic"),
        ("Straight Outta Compton", 2015, 7.8, "Biography Drama Music",
         "N.W.A, Compton, rap, Eazy-E, Dr. Dre, Ice Cube, hip-hop"),
        ("The Social Network", 2010, 7.7, "Biography Drama",
         "Facebook, Zuckerberg, Harvard, lawsuits, Sean Parker, algorithms"),
        ("Steve Jobs", 2015, 7.2, "Biography Drama",
         "Apple, iPhone, Wozniak, Sculley, keynote, perfectionism"),
        ("The Big Short", 2015, 7.8, "Biography Comedy Drama",
         "2008 crisis, mortgage, Wall Street, short selling, CDO, Ryan Gosling"),
        ("Wolf of Wall Street", 2013, 8.2, "Biography Comedy Crime",
         "Jordan Belfort, stockbroker, drugs, fraud, DiCaprio, Scorsese"),
        ("American Hustle", 2013, 7.3, "Crime Drama",
         "FBI sting, 1970s, New Jersey, cons, hair, Jennifer Lawrence"),
        ("Catch Me If You Can", 2002, 8.1, "Biography Crime Drama",
         "Frank Abagnale, con artist, impersonation, FBI, DiCaprio, Hanks"),
        ("Ocean's Eleven", 2001, 7.7, "Comedy Crime Thriller",
         "Vegas, heist, casino, Clooney, Pitt, ensemble, Soderbergh"),
        ("Now You See Me", 2013, 7.3, "Action Crime Mystery Thriller",
         "magicians, FBI, heist, Four Horsemen, illusion, twist"),
    ]
    rows_movies, rows_credits = [], []
    for i, (title, year, rating, genres, keywords) in enumerate(demo_movies, 1):
        genre_list = [{"id": j+10, "name": g} for j, g in enumerate(genres.split())]
        kw_list    = [{"id": j+100, "name": k} for j, k in enumerate(keywords.split(", "))]
        rows_movies.append({
            "id":           i,
            "title":        title,
            "genres":       str(genre_list),
            "keywords":     str(kw_list),
            "vote_average": rating,
            "overview":     keywords,
            "release_date": f"{year}-01-01",
        })
        # minimal credits — director = "Various"
        crew  = [{"job": "Director", "name": f"Director {i}"}]
        cast  = [{"name": f"Actor {j}", "character": "char"} for j in range(5)]
        rows_credits.append({
            "movie_id": i,
            "title":    title,
            "cast":     str(cast),
            "crew":     str(crew),
        })

    os.makedirs(DATA_DIR, exist_ok=True)
    pd.DataFrame(rows_movies).to_csv(MOVIES_CSV, index=False)
    pd.DataFrame(rows_credits).to_csv(CREDITS_CSV, index=False)
    print(f"✓ Demo dataset created ({len(demo_movies)} movies)")

# ── Feature building ──────────────────────────────────────────────
def build_features(movies, credits):
    df = movies.merge(credits, on="title")

    df["genres"]   = df["genres"].apply(lambda x: extract_names(x))
    df["keywords"] = df["keywords"].apply(lambda x: extract_names(x))
    df["cast"]     = df["cast"].apply(lambda x: extract_names(x, n=5))
    df["director"] = df["crew"].apply(extract_director) if "crew" in df.columns else [[]] * len(df)

    # slugify to avoid "Iron Man" vs "IronMan" collisions
    for col in ["genres", "keywords", "cast", "director"]:
        df[col] = df[col].apply(lambda lst: [slugify(x) for x in lst])

    df["tags"] = (
        df["genres"]   +
        df["keywords"] +
        df["cast"]     +
        df["director"]
    ).apply(lambda x: " ".join(x))

    df["tags"] = df["tags"].apply(stem_tokens)

    # keep movie_id if present
    if "id" in df.columns:
        df = df.rename(columns={"id": "movie_id"})

    return df[["movie_id", "title", "tags"]].reset_index(drop=True) if "movie_id" in df.columns \
        else df[["title", "tags"]].reset_index(drop=True)

# ── Train ─────────────────────────────────────────────────────────
def train():
    ensure_data()
    print("Loading CSVs …")
    movies  = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)

    print("Building feature matrix …")
    df = build_features(movies, credits)
    print(f"  {len(df)} movies processed.")

    print("Fitting TF-IDF (max_features=5000) …")
    tfidf   = TfidfVectorizer(max_features=5000, stop_words="english")
    vectors = tfidf.fit_transform(df["tags"]).toarray()

    print("Computing cosine similarity matrix …")
    sim = cosine_similarity(vectors)

    print("Saving model artifacts …")
    with open(os.path.join(MODEL_DIR, "movies.pkl"), "wb") as f:
        pickle.dump(df, f)
    with open(os.path.join(MODEL_DIR, "similarity.pkl"), "wb") as f:
        pickle.dump(sim, f)
    with open(os.path.join(MODEL_DIR, "tfidf.pkl"), "wb") as f:
        pickle.dump(tfidf, f)

    print(f"✓ Model saved to {MODEL_DIR}/")
    print(f"  movies.pkl     — {len(df)} rows")
    print(f"  similarity.pkl — {sim.shape} matrix")
    print("\nDone! Start the app with:  python app.py")

if __name__ == "__main__":
    train()
