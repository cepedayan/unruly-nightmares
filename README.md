# unruly-nightmares
Using NLP techniques and LLMs to mimic style and conversation
## Synopsis
This project is an exploration of Natural Language Processing techniques, pretrained large language models, and how they can be used to mimic writing style and interact with a literary work.
Here’s what the code does:
* Have a “conversation” with the poet Rubén Darío through his book Autobiografía ([Retrieval-Augmented Generation](https://github.com/cepedayan/unruly-nightmares/blob/main/retrieval_augmented_generation/Retrieval_Augmented_Generation.ipynb)).
* Find the most similar sentence in any given book and use it to mimic conversation ([Sentence Similarity](https://github.com/cepedayan/unruly-nightmares/blob/main/sentence_similarity/Sentence_Similarity.ipynb)).
* Replicate some of Gustavo Adolfo Becquer’s most prominent stylistic attributes ([Finetuning & encoding methods for text generation](https://github.com/cepedayan/unruly-nightmares/tree/main/finetuning)).

This project was developed for the KaggleX program. Special thanks to my mentor, Jonathan Schlosser, who not only guided me with their expertise in NLP, but also changed my perspective regarding my career in Data Science and had a tremendous impact on my own personal outlook and capabilities.

Although a lot more can be done —measure quality and compare model performance with linguistic KPIs, prompt engineering, experiment with number of epochs and track training loss—, this is the start of my journey with NLP.

### Example outputs
If you prefer to skip directly to the results, see here #pending.

## My experience
My starting point was a basic knowledge of Python and NLP concepts, and a very keen interest in understanding this marvelous technology. I wanted to learn how LLMs understand natural language, how they process it, and discover practical applications in creative spaces.

As a Localization Editor for Spanish in the Games industry, I seek to understand how LLMs can aid creative roles such as writers, translators, and editors in their daily tasks, in particular, those that relate to maintaining the voice and style of the brand and ensuring the continuity of characters and lore as valuable intellectual property of a company.
The questions that prompted this project were:
* How can LLMs learn stylistic attributes present in reference documents and replicate them in newly generated text?
* Can style be measured quantitatively with the help of an LLM, removing the editor’s subjective evaluation from the equation?
* What are the techniques used to make LLMs learn style vs learn facts present in any given document?

## The datasets
The following books were downloaded, cleaned, and preprocessed for using in various NLP techniques. (Public domain for their use in the US, gutemberg.org) 
* _Leyendas, cuentos y poemas_, by Gustavo Adolfo Bécquer
* _Obras escogidas_, by Gustavo Adolfo Bécquer
* _Autobiografía_, by Rubén Darío
  
## What I learned:
* About style ([See code here](https://github.com/cepedayan/unruly-nightmares/tree/main/finetuning)):
  * Stylistic attributes can be replicated by finetuning with a dataset in the target writing style. Attributes such as replication of imagery, specific typography not common in modern languages, average sentence length, register and tone, can all be learned by an open-source model with less than 1M characters.
  * The output can be greatly improved using various decoding methods—the best performance in this case was obtained with Beam search and Sampling.
    * I used DeepESP/gpt2-spanish, available in Hugging Face, and a dataset of ~750K characters.
* About having a “conversation” with a poet through their work:
  * Models that are pretrained for the Sentence Similarity task can be used to retrieve the source text that is most similar to the sentence fed in the prompt. This by no means resemble a conversation, but it’s a starting point. ([See code here](https://github.com/cepedayan/unruly-nightmares/tree/main/sentence_similarity))
  * Retrieval-Augmented Generation is a really powerful way to answer questions based on documents that are specific to any given task or industry. In combination with a powerful LLM, such as gpt-3.5-turbo, we get a really cool text generator, conversation partner, assistant—whatever we need, just give the instruction in the prompt!— with access to our knowledge repository. ([See code here](https://github.com/cepedayan/unruly-nightmares/tree/main/retrieval_augmented_generation))
    * Although we should beware of hallucinations—this can be somewhat offset by changing the temperature—, the advantage of RAG is that it allows us to track answers back to their original source (if present in the source documents). This is an advantage vs using the typical LLM chat platforms. The cost of embedding source documents and queries is an important consideration that should be weighed against the increase in productivity and resources required.
* About myself:
  * I possess the ability to ride steep learning curves—it’s quite amazing to think that the knowledge I started with was a vague idea of what a token was and that ChatGPT was a LLM.
  * Resilience is the biggest asset in the face of complicated problems—my ability to find answers and alternative solutions was put to the test after so many errors I had to solve. (See the error log here for your amusement.) #pending
  * NLP requires a combination of creative and logical approach to problems—my preferred type of challenge.
  * I learned to value my skills over the knowledge I have at any given time—problem-solving and communication is crucial.

## Libraries and techniques
Python libraries:
* numpy
* operating system (os)
* regex (re)
* scikit-learn

NLP techniques:
* Tokenization
* Finetuning
* Embeddings
* Decoding methods for text generation
* Prompt engineering
* Retrieval-augmented generation

NLP libraries and tools:
* Transformers (Hugging Face)
* chromadb

Deep learning and LLM frameworks:
* Torch
* TensorFlow
* LangChain
* OpenAI API

Others:
* Character encoding standards

## Output examples

These are some of the resulting "conversations" with each of the models and NLP techniques used.

### Finetuning (gpt2-spanish finetuned)
Prompt:
`"Será verdad que..."`

Output:
`Será verdad que no se ha de
hablar de la verdad, y que no se ha de
haber de la verdad, y que no se ha de
haber de la verdad, sino de la verdad, y que
se ha de creer que no se ha de creer en nada, y que
se ha de creer que no se ha de creer en nada, y que
se ha de creer en algo, y que no se ha de creer en nada, y que
se ha de creer en algo, y que no se ha de creer en nada, y que
se ha de creer en algo, y que no se ha de creer en nada, y que
se ha de creer en algo, y que se`

### Decoding methods (Sampling, gpt2-spanish finetuned)
Model output:

`Mientras flotaban en el aire la luz de la luna. La brisa comenzaba á refrescar el silencio; los árboles se ponían en movimiento. Los pájaros cantaban, á la luz de las lámparas, el viento de un
recuerdo sobre las hojas que se movían bajo sus pies. Entre tanto, el viento se había hecho más
poco en el`

Fragment from original book for comparison purposes:

`De todos modos, allá van estas cuartillas, valgan por lo
que valieren: que si alguien de más conocimientos é importancia,
una vez apuntada la idea, la desarrolla y prepara la opinión para
que fructifique, no serán perdidas del todo. Yo, entre tanto, voy á
trazar un tipo bastante original y que desconfío de poder reproducir.
Ya que no de otro modo, y aunque poco valga, contribuiré al éxito de
la predicación con el ejemplo.`

### Sentence Similarity (google/flan-t5/xxl)
Human.- `¿Cómo se llama la persona más agradable que conoce?`
Autobiografía.txt.- `Era un amable y jovial filósofo.`

Human.- `¿Cómo se llamaba?`
Autobiografía.txt.- `Se llamaba con un nombre balzaciano, Sebastián Menmolth.`

Human.- `¿Cómo describiría su personalidad?`
Autobiografía.txt.- `Escribía una prosa profusa, llena de vitalidad y de color, de plasticidad y de música.`

Human.- `¿Qué le gusta hacer a usted, además de la escritura?`
Autobiografía.txt.- `Extraña y ardua mezcla de cosas para la cabeza de un niño.`

Human.- `¿Me puede dar un ejemplo?`
Autobiografía.txt.- `¿Para dónde? Para cualquier parte.`

### Retrieval-Augmented Generation (gpt-3.5-turbo)

Emoji coding:
  ⛄ `temperature = 0.7`
  
  🥵 `temperature = 1`
  
  ✅ Correct answer, cross-checked with source document
  
  ❌ Incorrect answer, cross-checked with source document

Human.-	`Buen día, ¿me permite un minuto para hacerle unas preguntas?`
⛄ gpt-3.5-turbo.-	`Buen día, estimado interlocutor. Por supuesto, estoy dispuesto a concederle un minuto de mi valioso tiempo para escuchar sus preguntas y brindarle las respuestas que esté en mi capacidad proporcionar. Adelante, estoy todo oídos.`

Human.-	`¿Qué reuniones literarias era las que más disfrutaba?`
⛄ gpt-3.5-turbo.-	`Las reuniones literarias que más disfrutaba eran las que se llevaban a cabo en la casa de don Rafael Obligado.` ✅

Human.-	`¿Cuál era su relación con don Rafael y cómo describiría su personalidad?`
⛄ gpt-3.5-turbo.-	`Mi relación con don Rafael era de amistad y admiración mutua. Don Rafael era un hombre de gran carácter y determinación. Era un líder político apasionado y comprometido con sus ideales. Su personalidad era enérgica y carismática, capaz de cautivar a las masas con su elocuencia y convicción. Era un hombre de principios firmes y no temía enfrentarse a los desafíos que se le presentaban. Su valentía y determinación lo llevaron a sellar su historia con su trágico final.` ❌

Human.-	`¿Qué representa la poesía para usted?`
🥵 gpt-3.5-turbo.-	`La poesía, para mí, es como el susurro del viento entre las hojas de los árboles, es la melodía de los pájaros al amanecer y el aroma embriagador de las rosas en plena floración. Es el lenguaje de los sentimientos profundos y de las emociones indescriptibles. Representa una ventana hacia lo inefable, una forma de expresar lo inexpresable. La poesía es un bálsamo para el alma, un refugio en medio del caos, un remanso de paz en el agitado mar de la vida. ¿Y para usted, qué representa la poesía?` ✅ (At least, the answer resembles the imagery present in the source document: `Poética, sí, para quien sepa comprender el vaho de arte que flota sobre ese inconmesurable océano de tierra, sobre todo en los crepúsculos vespertinos y en los amaneceres.`)
⛄ gpt-3.5-turbo.-	`La poesía para mí representa una ventana hacia la belleza y la profundidad de las emociones humanas. Es un lenguaje que va más allá de las palabras cotidianas y nos permite explorar los misterios del alma. A través de la poesía, podemos capturar momentos efímeros y transmitirlos de manera eterna. Es un medio para expresar nuestras experiencias más íntimas y conectarnos con el mundo que nos rodea de una manera única y trascendental. ¿Cuál es tu poema favorito y por qué?` ❌

Human.-	`Adoro 'Los motivos del lobo', lo aprendí para recitarlo en la escuela.`
⛄ gpt-3.5-turbo.-	`Me alegra saber que también te gusta "Los motivos del lobo". Es una obra maravillosa de José Emilio Pacheco. Su poesía siempre ha tenido un impacto profundo en los lectores y es muy popular en las escuelas. ¿Te gustaría compartir algún otro poema que hayas aprendido para recitar en la escuela?` ❌ 😑
🥵 gpt-3.5-turbo.-	`Qué maravilloso que hayas aprendido "Los motivos del lobo" para recitarlo en la escuela. Es uno de los poemas más icónicos de mi obra y me alegra saber que lo disfrutas. ¿Hay algún otro poema que te gustaría aprender o recitar?` ✅ 😵

## Error log
#pending

