import { OpenAIEmbeddings } from "@langchain/openai";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { Document } from "langchain/document";

interface Split {
  pageContent: string;
  metadata: {
    prefix: string;
    number: string;
    [key: string]: any;
  };
}

interface ChapterRange {
  chapter: number;
  start: number;
  end: number;
}

// Save to vector store
await processPdfFile();

async function processPdfFile(): Promise<HNSWLib> {
  // Loading 
  const loader = new PDFLoader("./data/nist1.pdf");
  const docs = await loader.load();

  // Splitting and processing. Adds ONLY requirements to vector store
  const splitsWithMetadata: Split[] = [];

  for (const doc of docs) {
    const splits = customSplitter(doc.pageContent);
    
    splits.forEach(split => {
      split.metadata = {
        ...split.metadata,
        ...doc.metadata,
        chapter: getChapter(doc.metadata.loc.pageNumber)
      };
    });

    splitsWithMetadata.push(...splits);
  }

  // Embedding
  const embeddings = new OpenAIEmbeddings({ openAIApiKey: process.env.OPENAI_API_KEY });

  // Storing embeddings in vector store
  const vectorstore = await HNSWLib.fromDocuments(
    splitsWithMetadata as Document[],
    embeddings
  );

  await vectorstore.save("./vectorstore");

  return vectorstore;
}

function customSplitter(text: string): Split[] {
  const regex = /(PR:|PA:|FR:|PF:)(\d+\.\d+\s+.*?)(?=(PR:|PA:|FR:|PF:)|$)/gs;
  const matches = [...text.matchAll(regex)];
  
  return matches.map(match => ({
    pageContent: match[0].trim(),
    metadata: {
      prefix: match[1].slice(0, -1),
      number: match[2].split(' ')[0]
    }
  }));
}

function getChapter(pageNumber: number): number | null {
  const chapterRanges: ChapterRange[] = [
    { chapter: 0, start: 1, end: 10 },
    { chapter: 1, start: 11, end: 15 },
    { chapter: 2, start: 16, end: 22 },
    { chapter: 3, start: 23, end: 27 },
    { chapter: 4, start: 28, end: 44 },
    { chapter: 5, start: 45, end: 46 },
    { chapter: 6, start: 47, end: 85 },
    { chapter: 7, start: 86, end: 90 },
    { chapter: 8, start: 91, end: 102 },
    { chapter: 9, start: 104, end: 112 },
    { chapter: 10, start: 113, end: 121 },
    { chapter: 11, start: 122, end: 128 },
    { chapter: 12, start: 129, end: 130 },
    { chapter: 13, start: 131, end: 134 },
    { chapter: 14, start: 135, end: 147 }
  ];

  for (const range of chapterRanges) {
    if (pageNumber >= range.start && pageNumber <= range.end) {
      return range.chapter;
    }
  }

  return null;
}