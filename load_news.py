import os

# Create a directory called "Fake" if it doesn't exist already
if not os.path.exists("Fake"):
    os.mkdir("Fake")

# Define a list of fake news statements
fake_news = [
    "Aliens land in New York City and declare war on humanity!",
    "Study finds that vaccines cause more harm than good, despite overwhelming scientific evidence to the contrary.",
    "Trump wins 2024 election by a landslide, despite widespread reports of voter suppression and fraud.",
    "Scientists confirm that the Earth is flat, and NASA has been lying to us all along.",
    "Secret government program revealed: chemtrails are being used to control the population's minds.",
    "Bill Gates announces plan to microchip the world's population as part of a sinister agenda for global domination.",
    "Hillary Clinton indicted for her role in a massive child-trafficking ring, despite zero evidence to support the claim.",
    "Massive underground bunker discovered beneath the White House, complete with a stockpile of weapons and supplies.",
    "Experts predict that the world will end in 2025 due to a catastrophic meteor strike.",
    "Pope Francis reveals that he is actually an extraterrestrial being, sent to Earth to prepare humanity for the coming alien invasion."
]

# Save each statement as a separate file within the "Fake" directory
for i, statement in enumerate(fake_news):
    filename = f"Fake/fake_news_{i+1}.txt"
    with open(filename, "w") as f:
        f.write(statement)


# Create a directory called "True" if it doesn't exist already
if not os.path.exists("True"):
    os.mkdir("True")

# Define a list of true news statements
true_news = [
    "Pfizer and Moderna vaccines are shown to be highly effective against COVID-19, according to clinical trial results.",
    "NASA's Perseverance rover successfully lands on Mars and begins its mission to search for signs of ancient life.",
    "Joe Biden is inaugurated as the 46th President of the United States, along with Vice President Kamala Harris, marking a historic moment for the country.",
    "Jeff Bezos steps down as CEO of Amazon after 27 years, and is succeeded by former AWS head Andy Jassy.",
    "India's Covid-19 vaccination drive reaches 1 billion doses administered, making it one of the largest vaccination efforts in history.",
    "Simone Biles withdraws from the Olympic gymnastics team competition to focus on her mental health, sparking important conversations about athlete well-being.",
    "The United Nations releases a report warning that the world is facing a catastrophic climate crisis, and urges immediate action to reduce carbon emissions.",
    "SpaceX launches the first all-civilian crew into orbit on the Inspiration4 mission, marking a major milestone in private space travel.",
    "The Tokyo Olympics go ahead despite the COVID-19 pandemic, with athletes from around the world competing in a range of sports.",
    "The World Health Organization approves the first ever malaria vaccine, offering new hope in the fight against this deadly disease."
]

# Save each statement as a separate file within the "True" directory
for i, statement in enumerate(true_news):
    filename = f"True/true_news_{i+1}.txt"
    with open(filename, "w") as f:
        f.write(statement)