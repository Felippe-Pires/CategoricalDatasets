import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class CSV {
	
	public String algorithm;
	
	public CSV() {
		this.algorithm = "Zero++";
	}
	
	public String escapeSpecialCharacters(String data) {
    	if (data == null) {
    		throw new IllegalArgumentException("Input data cannot be null");
    	}
    	String escapedData = data.replaceAll("\\R", " ");
    	if (data.contains(";") || data.contains("\"") || data.contains("'")) {
    		data = data.replace("\"", "\"\"");
    		escapedData = "\"" + data + "\"";
    	}
    	return escapedData;
    }
    
    public String convertToCSV(String[] data) {
    	return Stream.of(data)
    			.map(this::escapeSpecialCharacters)
    			.collect(Collectors.joining(";"));
    }
    
    public void save_results(String path, String filename, List<String[]> dataLines) throws IOException {
		Path paths = Paths.get(path + "\\" + this.algorithm);
    	if (!Files.exists(paths))
    		Files.createDirectories(paths);
    	File csvOutputFile = new File(path + "\\" + this.algorithm + "\\" + filename);
    	try (PrintWriter pw = new PrintWriter(csvOutputFile)) {
    		dataLines.stream()
    		.map(this::convertToCSV)
    		.forEach(pw::println);
    	}
    }

}
