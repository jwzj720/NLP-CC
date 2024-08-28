import discord                       
from spacy.lang.en import English    # relevant for lab 3

nlp_model = English() # this makes an English spaCy model

test_str = """

CP341	
HomeSyllabusBlock ScheduleLabs	
Lab 1: Warmup
Since this is the beginning of the course and the year, we're doing a smaller lab (still worth full points!) to get going. Note that this used to be called Lab0 before the course was adjusted to the block plan, so some file names may reflect that. 

Starter Code
Get the starter code here! 


Setting Up (6 pts)
Your answers for this portion must be written in a pdf called Lab0.pdf made from this Overleaf LaTeX template. Follow the instructions here to copy this template so that you will have your own editable copy. This may seem cumbersome, but it is much easier than the previous method of making a pdf in this course, and it will familiarize you with Overleaf, which is an invaluable tool in academia. Every single research paper/book/article/etc. that I have ever contributed to was written on Overleaf. 

There’s a Docker image available for this course. See the course Docker instructions for details about how to work on your own machine.

To save yourself time for future weeks, I'd like you to make sure you can do the following two things on the command line. Put the requested response for each piece in the appropriate part of your pdf. You'll get 1 point just for submitting the PDF.


Navigate to the data directory /cs/cs159/data/gutenberg/. We can run the head command on that file to print out the first lines of the file.  e.g.:

$ head carroll-alice.txt

[Alice's Adventures in Wonderland by Lewis Carroll 1865]



CHAPTER I. Down the Rabbit-Hole



Alice was beginning to get very tired of sitting by her sister on the

bank, and of having nothing to do: once or twice she had peeped into the

book her sister was reading, but it had no pictures or conversations in

it, 'and what is the use of a book,' thought Alice 'without pictures or

conversation?'



Please pick a file in the directory other than carroll-alice.txt and copy the output of the head command into your pdf. (2 pts)

Run python3 to activate the interpreter. In the interpreter, run the following commands and verify that they don't throw any errors: 

>> import discord                       

>> from spacy.lang.en import English    # relevant for lab 3

>> nlp_model = English() # this makes an English spaCy model

Please verify that both of these import without errors. If there is any import error, run the following commands in the terminal to install the package. 

>> pip3 install <package name> --user   # For any package                   

>> pip3 install discord.py --user       # For discord

While we're not using spaCy for a while, we're going to do a quick test to see one of the cool things that spaCy can do: pull out the pieces of text it thinks are URLs! Try pasting in a snippet of text and seeing what it does with the output. I tried this out using the text from the homepage of a different course that I've taught:

>>> test_str = 

... [stuff I pasted in here]

... 

I then ran the following code to get a spaCy-processed version of my string split into pieces (we'll talk more about how it gets these pieces soon). We can check which pieces it thinks are URLs using the like_url attribute of each piece:

>>> spacy_output = nlp_model(test_str)

>>> for piece in spacy_output:

...     if piece.like_url:

...         print(piece)

https://www.gradescope.com/courses/...

Paste your input and the output you get into the corresponding section of your pdf. Quickly comment: how well did the code match URLs as compared to your expectations? If there are examples of things you thought it would get right or wrong where it behaved differently, mention them here. (3 pts)

If you can do these things, you should be set to go for upcoming labs. If you're getting stuck, please reach out to me to ask for help!

RegEx Practice (19 pts)
Practice regular expressions by playing RegEx Golf.

You must complete the following: (14 points)

Warmup (2pts)

Anchors (2pts)

It never ends (2pts)

Ranges (2pts)

Backrefs (2pts)

One additional puzzle of your choosing (4pts for the first puzzle, 1pt extra credit for each additional solution).

Screenshot of the RegEx Gold tool, showing words to match (which all contain "foo") and words not to match (none of which contain "foo").
Turn In
You will turn in a pdf called Lab0.pdf with your text answers and the file golf.py with your regex answers.  

In the file golf.py, fill in the empty strings with your best (that is, shortest) regular expression for each of the puzzles that you solved. You'll get full points as long as your regular expression didn't avoid the challenge of writing rules that generalize properties of the words (e.g. r"^(word1|word2|word3...)$" would be ridiculously long and a little silly). That said, if your length seems much higher than what's on the leaderboards, challenge yourself to get one that's shorter! Note that some people have used javascript to break the leaderboards and give themselves impossibly low scores (like 0 characters). (You will get 1 pt for a file with any edits, even if the regexs don't work. Since there are tiny differences between how the web application works and how Python regular expressions work, there's a small chance you may pass the test in the game but not pass the same test in a Python environment if this happens, take a crack at finding a solution that works in both environments, and come show your solution to me if you're having trouble doing that. Note that the Python grading code doesn't like aliases for sets (e.g., \w)!)

Additionally, in your pdf under Notes, include a few sentences describing how this activity went for you. Were any puzzles particularly challenging, and what made them challenging? Alternately, is there any regex you're particularly proud of devising, and if so, why? (4 pts)

When you are finished, submit both your Python file and your pdf.

Integrity Note
Since these puzzles are online, there are (undoubtedly) solutions posted somewhere. Under the honor code, I trust you to submit the best solutions you can come up with on your own.

NLP. Course materials from Julie Medero, Richard Wicentowski, Xanda Schofield, and Blake Jackson.

"""

spacy_output = nlp_model(test_str)

for piece in spacy_output:
    if piece.like_url:
        print(piece)
    