import java.io.IOException;
import java.io.StringReader;
import java.io.BufferedReader;
import java.util.List;
import java.util.ArrayList;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.io.PrintStream;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.Sentence;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.process.PTBTokenizer;

public class TokenizerSplitter {
    public static void main(String[] args) throws IOException {
        Path filePath = Paths.get(args[0]);
        try (BufferedReader br = Files.newBufferedReader(filePath, StandardCharsets.UTF_8);
              PrintStream out = new PrintStream(System.out, true, "UTF-8")) {
            String line;
            while ((line = br.readLine()) != null) {
                StringReader reader = new StringReader(line);
                DocumentPreprocessor dp = new DocumentPreprocessor(reader);
                List<String> sentenceList = new ArrayList<String>();

                for (List<HasWord> sentence : dp) {
                    String sentenceString = Sentence.listToString(sentence);
                    sentenceList.add(sentenceString.toString());
                }

                if (sentenceList.isEmpty()) {
                    sentenceList.add("");
                }

                out.println(sentenceList.size());
                for (String sentence : sentenceList) {
                    out.println(sentence);
                }
            }
        }
    }
}
