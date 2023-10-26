# unruly-nightmares
Using NLP techniques and LLMs to mimic style and conversation
## Synopsis
This project is an exploration of Natural Language Processing techniques, pretrained large language models, and how they can be used to mimic writing style and interact with a literary work.
Here’s what the code does:
* Have a “conversation” with the poet Rubén Darío through his book _Autobiografía_ ([Retrieval-Augmented Generation](https://github.com/cepedayan/unruly-nightmares/blob/main/retrieval_augmented_generation/Retrieval_Augmented_Generation.ipynb)).
* Find the most similar sentence in any given book and use it to mimic conversation ([Sentence Similarity](https://github.com/cepedayan/unruly-nightmares/blob/main/sentence_similarity/Sentence_Similarity.ipynb)).
* Replicate some of Gustavo Adolfo Becquer’s most prominent stylistic attributes ([Finetuning & encoding methods for text generation](https://github.com/cepedayan/unruly-nightmares/tree/main/finetuning)).

This project was developed for the KaggleX program. Special thanks to my mentor, Jonathan Schlosser, who not only guided me with their expertise in NLP, but also changed my perspective regarding my career in Data Science and had a huge impact on my own personal outlook and capabilities.

Although a lot more can be done —measure quality and compare model performance with linguistic KPIs, prompt engineering, experiment with number of epochs and track training loss—, this is the start of my journey with NLP.

### Output examples
If you prefer to skip directly to the results, [see here](#output-examples).

## My experience
My starting point was a basic knowledge of Python and NLP concepts, and a very keen interest in understanding this marvelous technology. I wanted to learn how LLMs understand natural language, how they process it, and discover practical applications in creative spaces.

As a Localization Editor for Spanish in the Games industry, I seek to understand how LLMs can aid creative roles such as writers, translators, and editors in their daily tasks, in particular, those that relate to maintaining the voice and style of the brand and ensuring the continuity of characters and lore as valuable intellectual property of a company.
The questions that prompted this project were:
* How can LLMs learn stylistic attributes present in reference documents and replicate them in newly generated text?
* Can style be measured quantitatively with the help of an LLM, removing the editor’s subjective evaluation from the equation?
* What are the techniques used to make LLMs learn style vs learn facts present in any given document?

## The datasets
The following books were downloaded, cleaned, preprocessed, and used as input to various NLP techniques. (Public domain for their use in the US, gutemberg.org) 
* _Leyendas, cuentos y poemas_, by Gustavo Adolfo Bécquer
* _Obras escogidas_, by Gustavo Adolfo Bécquer
* _Autobiografía_, by Rubén Darío
  
## What I learned
* About mimicking style with LLMs:
  * Stylistic attributes can be replicated by finetuning an LLM ([code here](https://github.com/cepedayan/unruly-nightmares/tree/main/finetuning)) with a dataset in the target writing style. Attributes such as replication of imagery, specific typography not common in modern languages, average sentence length, register and tone, can all be learned by an open-source model with less than 1M characters.
  * The output can be greatly improved using various decoding methods—the best performance in this case was obtained with Beam search and Sampling.
    * I used DeepESP/gpt2-spanish, available in Hugging Face, and a dataset of ~750K characters.
* About having a “conversation” with a poet through their work:
  * Models that are pretrained for the Sentence Similarity task ([code here](https://github.com/cepedayan/unruly-nightmares/tree/main/sentence_similarity)) can be used to retrieve the source text that is most similar to the sentence fed in the prompt. This by no means resembles a conversation, but it’s a starting point.
  * Retrieval-Augmented Generation ([code here](https://github.com/cepedayan/unruly-nightmares/tree/main/retrieval_augmented_generation)) is a really powerful way to answer questions based on documents that are specific to any given task or industry. In combination with a powerful LLM, such as gpt-3.5-turbo, we get a really cool text generator, conversation partner, assistant—whatever we need, just give the instruction in the prompt!— with access to our knowledge repository.
    * Although we should beware of hallucinations—this can be somewhat offset by changing the temperature—, RAG allows us to track answers back to their original source; this is an advantage vs using the typical LLM chat platforms. The cost of embedding source documents and queries is an important consideration that should be weighed against the increase in productivity and resources required.
* About myself:
  * I possess the ability to ride steep learning curves—it’s amazing to think that all I knew about NLP at the start of this 3-month journey was a vague idea of what a token was and that ChatGPT was a LLM. :sweat_smile:
  * Resilience is the biggest asset in the face of complicated problems—my ability to find answers and alternative solutions was put to the test after so many errors I had to solve. (See the error log here for your amusement.) #pending
  * NLP requires a combination of creative and logical approach to problems—my preferred type of challenge.
  * I learned to value my skills over knowledge—knowledge can be acquired quickly, but problem-solving and communication is crucial for delivering results.

## Tools and techniques used
Python libraries:
* numpy
* operating system (os)
* regex (re)
* scikit-learn
* matplotlib

NLP & ML techniques:
* Tokenization
* Embeddings
* Finetuning
* Decoding methods for text generation
* Prompt engineering
* Retrieval-augmented generation
* Principal Component Analysis

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

These are some of the "conversations" resulting from each technique--the link to the code is included.

### [Finetuning - gpt2-spanish finetuned](https://github.com/cepedayan/unruly-nightmares/tree/main/finetuning)
The model was finetuned successfully, but it generates repetitive nonsense--no resemblance to the writing style of our author.
For improvements and fixes to the output, see [Decoding_methods-gpt2_finetuned.ipynb](https://github.com/cepedayan/unruly-nightmares/blob/main/finetuning/Decoding_Methods-gpt2_finetuned.ipynb), where I use different decoding methods for language generation with Transformers.

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

### [Decoding methods - Sampling, gpt2-spanish finetuned](https://github.com/cepedayan/unruly-nightmares/blob/main/finetuning/Decoding_Methods-gpt2_finetuned.ipynb)
Resemblance of the writing style of our author is present in the output generated by the finetuned model when using the following decoding methods:

*   Beam search (with n-gram penalties)
*   Top-K and Top-p sampling

Stylistic attributes that were replicated:
*   Use of vivid and descriptive language to help create mental images.
*   Old Spanish typography, like the accent mark in the preposition "á".
*   Longer sentences with detailed descriptions.
*   Formal register present in adverbial and verbal expressions.

Stylistic attributes that were not replicated:
*   Syntactic complexity--our model can create longer sentences, but lacks logical coherence.

Model output:

> `Mientras flotaban en el aire la luz de la luna. La brisa comenzaba á refrescar el silencio; los árboles se ponían en movimiento. Los pájaros cantaban, á la luz de las lámparas, el viento de un
recuerdo sobre las hojas que se movían bajo sus pies. Entre tanto, el viento se había hecho más
poco en el`

Fragment from original book for comparison purposes:

> `De todos modos, allá van estas cuartillas, valgan por lo
que valieren: que si alguien de más conocimientos é importancia,
una vez apuntada la idea, la desarrolla y prepara la opinión para
que fructifique, no serán perdidas del todo. Yo, entre tanto, voy á
trazar un tipo bastante original y que desconfío de poder reproducir.
Ya que no de otro modo, y aunque poco valga, contribuiré al éxito de
la predicación con el ejemplo.`

### [Sentence Similarity - google/flan-t5/xxl](https://github.com/cepedayan/unruly-nightmares/tree/main/sentence_similarity)
Sentence Similarity models can be used to "talk" to the document, the book _Autobiografía_ in this case, by retrieving the sentence with the highest cosine similarity in the vector space (embeddings).
On the other hand, Text2Text Generation models might not be suitable for open-ended text generation, depending on pretraining and finetuning.
  *   For our model "google/flan-t5-xxl" to output conversational text in the form of answers to questions like "How are you", we would need to finetune with a dataset that includes conversational interactions.
  *   The result is an interview more than a conversation, with stiff "answers" as they are presented verbatim from the source document.

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

### [Retrieval-Augmented Generation - gpt-3.5-turbo](https://github.com/cepedayan/unruly-nightmares/tree/main/retrieval_augmented_generation)
🥳 Success! By using RAG, we can "talk" to a poet through their work!
By embedding and retrieving relevant documents--the sentences in the book _Autobiografía_--, we can obtain answers from the LLM based on the content present in the book.
As we can see in the example below, hallucinations can somewhat be offset by changing the temperature. Example: When asked about one of the poet's most famous poems with `temperature = 0.7`, the LLM attributed the poem to a different author!🤯 By changing to `temperature = 1`, this fact was corrected.

The use of a **powerful LLM such as gpt-3.5-turbo** was beneficial in this project for the following reasons:

*   The ability to sustain a two-way conversation, by prompting instructions to interact as a conversational partner, such as providing answers that follow certain style and asking questions relevant to the topic.
*   Increased ability to understand and contextualize user inputs, providing more accurate and contextually relevant responses.

😨 An interesting observation about the last answer in the Output example below:
*   I did not prompt the LLM to assume Rubén Darío's persona, this fact is not explicit in the source documents, and it was never mentioned in the conversation--gpt-3.5-turbo inferred and assumed the character based on a global understanding of the book and associations learned during its pretraining (that's my guess, at least).

**Emoji coding:**

  ⛄ `temperature = 0.7`
  
  🥵 `temperature = 1`
  
  ✅ Correct answer, cross-checked with source document
  
  ❌ Incorrect answer, cross-checked with source document

**Conversation:**  

Human.-	`Buen día, ¿me permite un minuto para hacerle unas preguntas?`

gpt-3.5-turbo(⛄)	`Buen día, estimado interlocutor. Por supuesto, estoy dispuesto a concederle un minuto de mi valioso tiempo para escuchar sus preguntas y brindarle las respuestas que esté en mi capacidad proporcionar. Adelante, estoy todo oídos.`

Human.-	`¿Qué reuniones literarias era las que más disfrutaba?`

gpt-3.5-turbo(⛄)	`Las reuniones literarias que más disfrutaba eran las que se llevaban a cabo en la casa de don Rafael Obligado.` ✅

Human.-	`¿Cuál era su relación con don Rafael y cómo describiría su personalidad?`

gpt-3.5-turbo(⛄)	`Mi relación con don Rafael era de amistad y admiración mutua. Don Rafael era un hombre de gran carácter y determinación. Era un líder político apasionado y comprometido con sus ideales. Su personalidad era enérgica y carismática, capaz de cautivar a las masas con su elocuencia y convicción. Era un hombre de principios firmes y no temía enfrentarse a los desafíos que se le presentaban. Su valentía y determinación lo llevaron a sellar su historia con su trágico final.` ❌

Human.-	`¿Qué representa la poesía para usted?`

gpt-3.5-turbo(⛄)	`La poesía para mí representa una ventana hacia la belleza y la profundidad de las emociones humanas. Es un lenguaje que va más allá de las palabras cotidianas y nos permite explorar los misterios del alma. A través de la poesía, podemos capturar momentos efímeros y transmitirlos de manera eterna. Es un medio para expresar nuestras experiencias más íntimas y conectarnos con el mundo que nos rodea de una manera única y trascendental. ¿Cuál es tu poema favorito y por qué?` ❌

Human.-	`Adoro 'Los motivos del lobo', lo aprendí para recitarlo en la escuela.`

gpt-3.5-turbo(⛄)	`Me alegra saber que también te gusta "Los motivos del lobo". Es una obra maravillosa de José Emilio Pacheco. Su poesía siempre ha tenido un impacto profundo en los lectores y es muy popular en las escuelas. ¿Te gustaría compartir algún otro poema que hayas aprendido para recitar en la escuela?` ❌ 😑
gpt-3.5-turbo(🥵)	`Qué maravilloso que hayas aprendido "Los motivos del lobo" para recitarlo en la escuela. Es uno de los poemas más icónicos de mi obra y me alegra saber que lo disfrutas. ¿Hay algún otro poema que te gustaría aprender o recitar?` ✅ 😵

## Error log
#pending

