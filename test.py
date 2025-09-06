from transformers import pipeline
agent = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_new_tokens=int(2e6),
    temperature=0.1,
    do_sample=True,
    use_cache=False,
    # trust_remote_code=True  # Required for Phi-3
)

# prompt = "Correct this text: this is a test sentance with erors\nCorrected text:"
leading = "You are a English language expert and you have to correct the parsing errors made during OCR in the following text: "
target = "In theLond0n Journal, of March, 1732, is a curiovs, and, of course, credible account of a particular case of vampyrifin, which is stated to hove accurred at Madreyga, in Hungary. It appears, that upon an examination of the cornmander-in-chief arid magistrates of tbe place, they positively and unanimously affirmed, that, about five years before, a certairi Heyduke, named Arnold Paul, had bcen beclrd to say, that, at Cassovia, ori the fr0ntiers of the Turkish Servio, he had been tormented by a vampyre, but had found a way to rid himself of the euj1, by eating some of the earth out of the vampyre's grove, and rubbing himſelf with his blood. "
# This prccaution, however, did not prevent him from bccoming a vampyre himſels; sor, about twenty or thirty days after his death and burial, many persons complainod of hauing 6een tormented by him, and a deposition was made, that four persons had been deprived os life by his attacks. To prevent further mischief, the lnhabitants havjng consulted their Hadagni, took up the body, and f ound it (aſ is supposed to be usual in cafes of uampyrism) fresh, and entjrely free from corruptjon, and emitting at the rnouth, riose, and ears, pure and fIorid blood. Proof having been thus obtained, they resorted to the accustomed remedy. A stake was driven entirely lhrough the bearl and body of Arnold Paul, at which he is reported to hauecried out cls dreadfully as is he had been olive. This done, they cut ofs his head, burned his body, and threw lhe asbes into his grave. The same measures were adopted with the corses of th ose persons who had previously dicd from varnpyrism, lest they should, in theirturn, become clgentf upan others who survived them."
trailing = "\Corrections: "
prompt = leading + target + trailing
result = agent(prompt)