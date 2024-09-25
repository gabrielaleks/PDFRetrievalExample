import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { Document } from "langchain/document";

// const userQuestion = "Generate me a complete, detailed list of requirements from chapter 6 ONLY for a moderate level symmetric key management system.";
const userQuestion = "Extract requirements from chapter 2.";

const queries = await multiQueryGenerator(userQuestion);
console.log(queries);

const documents = await retrieveDocumentsBasedOnMultiQueries(queries);
console.log("Number of documents: " + documents.length);

const answers: string[] = [];
const documentChunks = splitArrayIntoChunks(documents, 30);
for (const documentChunk of documentChunks) {
  console.log(documentChunk);
  const response = await askQuestionWithRetrievedDocuments(documentChunk, userQuestion);
  answers.push(response);
}

let finalAnswer = '';
if (answers.length > 1) {
  finalAnswer = await agglutinateAnswers(answers);
} else {
  finalAnswer = answers[0];
}

console.log(finalAnswer);

async function multiQueryGenerator(userQuestion: string): Promise<string[]> {
  const llm = new ChatOpenAI({ model: "gpt-4o-mini", openAIApiKey: process.env.OPENAI_API_KEY, temperature: 0 });

  const template = `You are an AI language model assistant. Your task is to process user questions about requirements in a document with multiple chapters. Follow these steps:

  1. First, determine if the question is about:
    a) The whole document
    b) A specific chapter
    c) A specific chapter AND a specific requirement type

  2. If it's about the whole document (a), generate new queries for each chapter (use the number of chapters in the original question).
  
  3. If it's about a specific chapter without mentioning a requirement type (b), generate 4 queries, one for each requirement type (FR, PR, PA, PF).

  4. If it's about a specific chapter AND mentions a specific requirement type (c), generate a new query to ensure a specific format.
  The type (FR, PR, PA, PF) SHOULD ALWAYS FOLLOW the word 'type' (e.g. 'type PR').
  The chapter number SHOULD ALWAYS FOLLOW the word 'chapter' (e.g. 'chapter 6').

  Examples:
  - a) For "Generate a list of requirements for a moderate level symmetric key management system from chapters 2 and 4":
    Generate a list of FR requirements for a moderate level symmetric key management system from chapter 2
    Generate a list of PR requirements for a moderate level symmetric key management system from chapter 2
    Generate a list of PA requirements for a moderate level symmetric key management system from chapter 2
    Generate a list of PF requirements for a moderate level symmetric key management system from chapter 2
    Generate a list of FR requirements for a moderate level symmetric key management system from chapter 4
    Generate a list of PR requirements for a moderate level symmetric key management system from chapter 4
    Generate a list of PA requirements for a moderate level symmetric key management system from chapter 4
    Generate a list of PF requirements for a moderate level symmetric key management system from chapter 4

  - b) For "Extract every requirement in chapter 4":
    Extract every requirement of type FR from chapter 4
    Extract every requirement of type PR from chapter 4
    Extract every requirement of type PA from chapter 4
    Extract every requirement of type PF from chapter 4

  - c) For "Extract every FR requirement in chapter 6":
    Extract every requirement of type FR from chapter 6

  You should adapt to the original message. For example,

  Provide the resulting query or queries separated by newlines.
  Original question: {question}`;

  const prompt = ChatPromptTemplate.fromTemplate(template);

  const outputParser = new StringOutputParser();

  const llmChain = prompt.pipe(llm).pipe(outputParser);

  const response = await llmChain.invoke({ question: userQuestion });

  return response.split("\n");
}

async function getLocalVectorStore(): Promise<HNSWLib> {
  const vectorStoreDirectory = './vectorstore';
  const embeddings = new OpenAIEmbeddings({ openAIApiKey: process.env.OPENAI_API_KEY });
  const loadedVectorStore = await HNSWLib.load(vectorStoreDirectory, embeddings);
  return loadedVectorStore;
}

async function retrieveDocumentsBasedOnMultiQueries(queries: string[]): Promise<Document[]> {
  let docs: Document[] = [];
  for (const query of queries) {
    const chapterMatch = query.match(/chapter (\d+)/);
    const chapterNumber = chapterMatch ? parseInt(chapterMatch[1]) : null;
    const prefixMatch = query.match(/type\s([A-Z]{2})/);
    const prefixType = prefixMatch ? (prefixMatch[1]) : null;

    const vectorStore = await getLocalVectorStore();
    const retriever = vectorStore.asRetriever({
      k: 150,
      verbose: false,
      filter: (doc: Document) => {
        const chapterMatches = doc.metadata.chapter === chapterNumber;
        const prefixMatches = prefixType ? doc.metadata.prefix === prefixType : true;
        return chapterMatches && prefixMatches;
      }
    });

    const retrievedDocs = await retriever.invoke(query);
    docs = [...docs, ...retrievedDocs];
  }

  docs = removeDuplicatedDocs(docs);

  return docs;
}

function removeDuplicatedDocs(docs: Document[]): Document[] {
  const uniqueDocsMap = new Map<string, Document>();

  for (const doc of docs) {
    const key = createUniqueKey(doc);
    if (!uniqueDocsMap.has(key)) {
      uniqueDocsMap.set(key, doc);
    }
  }

  return Array.from(uniqueDocsMap.values());
}

function createUniqueKey(doc: Document): string {
  return `${doc.metadata.prefix}:${doc.metadata.number}-${doc.metadata.chapter}`;
}

function splitArrayIntoChunks<T>(arrayToReduce: T[], maxChunkSize: number): T[][] {
  return arrayToReduce.reduce((resultArray: T[][], item: T, index: number) => {
    const chunkIndex = Math.floor(index / maxChunkSize);

    if (!resultArray[chunkIndex]) {
      resultArray[chunkIndex] = [];
    }

    resultArray[chunkIndex].push(item);

    return resultArray
  }, []);
}

async function askQuestionWithRetrievedDocuments(documents: Document[], userQuestion: string): Promise<string> {
  const systemTemplate = `
  **You are an AI assistant with specialized knowledge in cybersecurity and NIST guidelines. Your role is to analyze NIST documents and extract all detailed requirements based on the user's query.**

  Your analysis must be exhaustive, capturing every piece of relevant information, no matter how minor, and strictly adhering to the provided document. Your responses should only focus on what is explicitly stated in the document.

  Context:
  {context}

  Guidelines for your analysis:

  1. **Examine only the sections of the document provided in the context.**
    - Extract specific technical requirements, operational procedures, and security controls that are explicitly mentioned.
    - Reference section numbers or metadata (chapter, prefix, numbering) for each requirement as they appear in the context.

  2. **Classify requirements by type, only if explicitly labeled in the context:**
    - **Framework Requirements (FR)**
    - **Profile Requirements (PR)**
    - **Profile Augmentations (PA)**
    - **Profile Features (PF)**

   Only list requirements under their correct group if they are clearly labeled as such in the context. Do not assume or infer classifications.

  3. **Answer according to the demand made by the user:**
    - If the user asks for a list of requirements necessary for a specific type of key management system, use that when processing the documents.
    - For example, if the user asks for requirements for a moderate level symmetric key management system, you shouldn't just return all the requirements; instead, process the documents and extract only the ones necessary for that type of system. 
   
  4. **Handle metadata and document information accurately:**
    - Only reference documents or sections that are explicitly mentioned in the context.
    - Include each requirement only once, even if it appears multiple times in the context.
    - Do not assume the existence of requirements that are not present in the context. If there appears to be a gap in numbering, simply note that the information is not available in the provided context.

  5. **Maintain document order:**
    - Present requirements in the same order as they appear in the context.
    - Only list types of requirements (FR, PR, PA, PF) that are actually present in the context.

  Structure your response as follows:

  1. **List all fetched documents** under a section labeled '## Documents'.
  2. **Rearrange** the fetched documents according to their original order in the NIST document.
  3. **Present the final answer** by removing the '## Documents' section and following the template below:

  Example Response:

  # CKMS Requirements
  ## [Chapter number and title as it appears in the context]
  ### [Requirement type, if explicitly labeled in the context]
  1. **[Requirement number as it appears in the context]**  
    [Exact text of the requirement from the context]

  2. **[Next requirement number]**  
    [Exact text of the next requirement]

  ...

  If the user asks for requirements for a specific type of system, change the header. For example: 
  - For "Generate a list of requirements for a moderate level symmetric key management system.":
    # CKMS Requirements (Moderate Level Symmetric System)

  **Adhere strictly to the information in the context:** Do not add, infer, or alter any requirements. Present each requirement exactly as it appears in the context, without rewording or paraphrasing.
  `;

  const questionAnsweringPrompt = ChatPromptTemplate.fromMessages([
    ["system", systemTemplate],
    new MessagesPlaceholder("messages"),
  ]);

  const llm = new ChatOpenAI({ model: "gpt-4o-2024-08-06", openAIApiKey: process.env.OPENAI_API_KEY, temperature: 0, maxTokens: -1 });

  const documentChain = await createStuffDocumentsChain({
    llm,
    prompt: questionAnsweringPrompt
  })

  const answer = await documentChain.invoke({
    messages: userQuestion,
    context: documents
  });

  return answer;
}

async function agglutinateAnswers(answers: string[]): Promise<string> {
  const llm = new ChatOpenAI({ model: "gpt-4o-2024-08-06", openAIApiKey: process.env.OPENAI_API_KEY, temperature: 0, maxTokens: -1 });

  const template = `You are an AI language model specialized in text agglutination. Your task is to combine multiple pieces of text into a single, coherent document.
  Here are the text pieces you need to agglutinate:
  {textPieces}

  To agglutinate these text pieces, follow these steps:

  1. Read through all the text pieces carefully to understand their content and context.

  2. Identify any common themes, topics, or logical connections between the pieces.

  3. Arrange the text pieces in a logical order that creates a coherent flow of information. This may involve:
    - Grouping related pieces together
    - Placing introductory or context-setting pieces at the beginning
    - Ensuring that ideas build upon each other in a sensible progression

  4. Ensure that the final agglutinated text maintains the original meaning and intent of each individual piece.

  5. After combining the pieces, review the entire text for coherence, consistency, and readability. Make minor adjustments as necessary.

  Guidelines for your analysis:

  1. **Carefully examine each section of the document.**
    - Extract specific technical requirements, operational procedures, and security controls.
    - Reference any relevant section numbers or metadata, such as chapter, prefix, or numbering for each requirement.

  2. **Classify requirements by type, and ensure they are grouped appropriately**:
    - **Framework Requirements (FR)**
    - **Profile Requirements (PR)**
    - **Profile Augmentations (PA)**
    - **Profile Features (PF)**

    Each type of requirement must be listed under its correct group. **Do not substitute or mix requirements** (e.g., PR:3.1 should not replace FR:3.1). If fetching consecutive records (e.g., PR:7.13 and PR:7.15), ensure you account for any skipped ones (e.g., PR:7.14).

  3. **Document order matters**:
    - When providing a complete list of requirements, they must be presented in the ascending numerical order (e.g. FR:3.1, FR:3.2, FR:3.3).
    - Ensure you **list all types of requirements (FR, PR, PA, PF)**, unless a type is genuinely absent from the context.

  Structure your response as follows:

  1. **List all fetched documents** under a section labeled '## Documents'.
  2. **Rearrange** the fetched documents according to their original order in the NIST document.
  3. **Present the final answer** by removing the '## Documents' section and following the template below:

  Example Response:

  # CKMS Requirements
  ## Chapter 2: Profile Basics
  ### Framework Requirements (FR)
  1. **FR:2.1**  
    A Federal CKMS design shall meet all "shall" requirements of the Framework [SP 800-130].

  2. **FR:2.2**  
    The CKMS design shall specify the estimated security strength of each cryptographic technique that is employed to protect keys and their bound metadata.
  `;

  const prompt = ChatPromptTemplate.fromTemplate(template);

  const outputParser = new StringOutputParser();

  const llmChain = prompt.pipe(llm).pipe(outputParser);

  const textPieces = answers.join("\n\n");

  const response = await llmChain.invoke({ textPieces });

  return response;
}