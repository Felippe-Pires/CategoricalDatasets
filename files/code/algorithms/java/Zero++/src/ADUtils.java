import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 *
 * @author Guansong Pang 14/02/2014
 *
 */

public class ADUtils {
    
    private static Instances testInstances;
    private static Instances trainInstances;
    public static String[] dataSetFullNameList;
    public static String[] dataSetNameList;
    public static String dataFilename;
    
    private static double AUC;
    private static double timeofBuilding = 0;
    private static double timeofTesting = 0;
    
    public static String t = "50";
    public static String n = "8";
    public static String m = "2";
    public static String runs = "10";
    
    public static String dir_result = "results";
    public static List<List<Double>> scores_runs = new ArrayList<>();
    
    public static void main(String[] args) throws Exception
    {
    	String paths = "datasets";
				
    	String[] path = paths.split(",");
    	for(String str: path) {
	        boolean dataFileIsDir = new File(str).isDirectory();
	        if(dataFileIsDir)
	            buildDataSetsPathList(str);
	        else
	            dataFilename = str;
	        //anomalyDetectorOptions();
	        anomalyDetectorOptions(dataFileIsDir);
    	}
    }
    
    /**
     * to invoke one anomaly detector
     * @param flag boolean parameter used to determine whether we are trying to detect anomalies in a data set or a folder of data sets.
     * @throws Exception
     */
    public static void anomalyDetectorOptions(boolean flag) throws Exception {
        
        if(flag) {
            for (int count = 0; count < dataSetFullNameList.length ; count++)
            {
                dataFilename = dataSetFullNameList[count];
                if (dataFilename != null) {
	                System.out.print(dataFilename.substring(dataFilename.lastIndexOf("\\")+1,dataFilename.lastIndexOf("."))+",");
	                // trainInstances = readDataSet(dataFilename);
	                testInstances = readDataSet(dataFilename);
	                // System.out.println(dataSetFullNameList[count]);
	                scores_runs = new ArrayList<>();
	                runZeroPlusPlus();
	                timeofBuilding = 0;
	                timeofTesting = 0;
                }
            }
        } else {
            System.out.println(dataFilename.substring(dataFilename.lastIndexOf("\\")+1)+",");
            trainInstances = readDataSet(dataFilename);
            testInstances = readDataSet(dataFilename);
            runZeroPlusPlus();
        }
    }
    
    /**
     * to invoke one anomaly detector
     * @throws Exception
     */
    public static void anomalyDetectorOptions() throws Exception {
        for (int count = 0; count < dataSetFullNameList.length ; count++)
        {
            dataFilename = dataSetFullNameList[count];
            System.out.println(dataFilename.substring(dataFilename.lastIndexOf("\\")+1,dataFilename.lastIndexOf("."))+",");
            //trainInstances = readDataSet(dataFilename);
            testInstances = readDataSet(dataFilename);
            runZeroPlusPlus();            
            timeofBuilding = 0;
            timeofTesting = 0;            
        }
    }
    
    public static void runZeroPlusPlus() {
        System.out.print("t:"+t+",n:"+n+","+"m:"+m+","+"runs:"+runs+",");
        Random r = new Random(1);
        int seed = 1;
        int run_num = Integer.valueOf(runs);
        double [] aucs = new double[run_num];
        for(int j = 0; j < run_num; j++) {
            trainInstances = readDataSet(dataFilename);
            seed = r.nextInt(trainInstances.numInstances());
            String[] options = getZeroPlusPlusSetOptions();
            options[1] = String.valueOf(t); // #subsamples
            options[3] = n; // subsampling size
            int dim = Integer.valueOf(m);
            zeroPlusPlus(options,seed,dim);            
            aucs[j] = AUC;
        }
        save_results();
        outputSTDandMEAN(aucs,run_num);
    }
    /**
     * to invoke Isolation Forest
     * @param options the parameters for isolation forest
     */
    public static void zeroPlusPlus(String[] options,int seed,int dim_num)
    {
        
        try
        {
            Zero zero = new Zero();
            zero.setOptions(options);
            zero.setSubspaceDim(dim_num);
            zero.setSelfGeneratedRandomSeed(seed);
            double startTime = System.currentTimeMillis();
            zero.buildClassifier(trainInstances);
            double endTime = System.currentTimeMillis();
            timeofBuilding += endTime - startTime;
            startTime = System.currentTimeMillis();
            Hashtable<Integer,Double> outlierScores = zero.scoringTestInstances(testInstances);
            addScores(outlierScores);
            endTime = System.currentTimeMillis();
            timeofTesting += endTime - startTime;
            // System.out.print(timeofTesting / 1000 +",");
            Instances classes = divide(testInstances,true);
            LinkedHashMap<Integer,Integer> rankList = rankInstancesBasedOutlierScores(outlierScores,classes);
            AUC = computeAUCAccordingtoOutlierRanking(classes,rankList);
            
        }
        catch(Exception e)
        {
            timeofBuilding = 0;
            timeofTesting = 0;
            AUC = -1;
        }
        
    }    
    
    public static void addScores(Hashtable<Integer,Double> scores) {
    	//Lista de scores
    	List<Double> numbersList = new ArrayList<>();
        Enumeration<Double> enumeration = scores.elements();
        while (enumeration.hasMoreElements()) {
            numbersList.add(enumeration.nextElement());
        }
        Collections.reverse(numbersList);
    	scores_runs.add(numbersList);
    }
    
    public static List<Integer> findIndexesOfValue(List<Double> list, double value) {
        List<Integer> indexes = new ArrayList<>();
        for (int i = 0; i < list.size(); i++) {
            if (list.get(i) == value) {
                indexes.add(i);
            }
        }
        return indexes;
    }
    
    public static int[] calculateRanking(List<Double> inputList) {
        int[] rankingList = new int[inputList.size()];
        List<Double> sortedList = new ArrayList<>(inputList);
        Collections.sort(sortedList, Collections.reverseOrder());

        int count = 1;
        for (int i=0 ; i<sortedList.size() ; i++) {
            List<Integer> rank = findIndexesOfValue(inputList, sortedList.get(i));
            for (Integer index : rank)
            	rankingList[index] = count++;
            i += rank.size()-1;
        }

        return rankingList;
    }
    
    public static String acertou(int num_outliers, String tipo, int position_ranking) {
    	if(tipo == "I" && position_ranking > num_outliers)
    		return "True";
    	if(tipo == "O" && position_ranking <= num_outliers)
			return "True";
    	return "False";
    }

    public static void save_results() {
    	List<String[]> dataLines = new ArrayList<>();
    	dataLines.add(new String[] {
    			"dataset", "algoritmo", "parameter", "point", "type", "detect", "score", "ranking"
    		});
    	
    	//M�dia dos scores
    	double[] avg_scores_ = new double[testInstances.numInstances()];
    	for (List<Double> run: scores_runs) {
    		for (int count=0 ; count < run.size() ; count++) {
    			avg_scores_[count] += run.get(count);
    		}
    	}
    	List<Double> avg_scores = new ArrayList<>();
    	for (int count=0 ; count < avg_scores_.length ; count++) {
    		avg_scores.add(avg_scores_[count] / scores_runs.size());
    	}
    		
    	int num_outliers = 0;
    	for(int idx = 0; idx < testInstances.numInstances(); idx++) {
    		String[] columns = testInstances.instance(idx).toString().split(",");
    		String tipo = (columns[columns.length-1].trim().equals("no")) ? "I" : "O";
    		if(tipo == "O")
				num_outliers++;
    	}
        
        //Ranking
        int[] ranking = calculateRanking(avg_scores);
    	
    	String[] tmp = dataFilename.split("\\\\");
    	String arquivo = tmp[tmp.length-1];
    	for(int idx = 0; idx < testInstances.numInstances(); idx++) {
    		String[] columns = testInstances.instance(idx).toString().split(",");
    		String tipo = (columns[columns.length-1].trim().equals("no")) ? "I" : "O";
    		dataLines.add(new String[] {
    			arquivo, "zero++", "", "" + (idx+1), tipo, acertou(num_outliers, tipo, ranking[idx]), avg_scores.get(idx).toString(), ""+ranking[idx]
    		});
    	}
    	
    	CSV csv = new CSV();
    	try {
			csv.save_results(dir_result, arquivo, dataLines);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
    
    /**
     * to set the parameters of isolation forest
     * @return return an option string array, which parameterizes isolation forest
     */
    public static String[] getZeroPlusPlusSetOptions() {
        String[] options = new String[6];
        options[0] = "-I";
        options[1] = "50";
        options[2] = "-N";
        options[3] = "8";
        options[4] = "-S";
        options[5] = "1";
        return options;
    }
    
    
    
    /**
     * to rank instances based on their outlier scores
     * @param outlierScores a hashtable with instances IDs as keys and outlier scores as values
     * @return the ranked instances according to the outlier scores
     */
    public static LinkedHashMap<Integer,Integer> rankInstancesBasedOutlierScores(Hashtable<Integer,Double> outlierScores, Instances classes) {
        
        List list = new ArrayList(outlierScores.entrySet());
        Collections.sort(list, new Comparator<Map.Entry>() {            
            
            @Override
            public int compare(Map.Entry o1, Map.Entry o2) {
                return ((Double) o2.getValue()).compareTo((Double) o1.getValue()); //descending ranking
                //return ((Double) o1.getValue()).compareTo((Double) o2.getValue()); // asending ranking
            }
            
        });
        LinkedHashMap<Integer,Integer> rankList = new LinkedHashMap();
        for(int i = 0; i < list.size(); i++) {
            Map.Entry object = (Map.Entry<Integer, Double>)list.get(i);
            int index = (Integer) object.getKey();
//            double score = (Double) object.getValue();
            rankList.put(index, i+1);           
        }
        return rankList;
    }
    
    /**
     * to calculate the AUC score based on the ranking of outlier scores
     * @param classes the class labels of instances
     * @param rankList the ranking list of instances w.r.t outlier scores
     * @return the AUC score
     */
    public static double computeAUCAccordingtoOutlierRanking(Instances classes, LinkedHashMap<Integer,Integer> rankList) {
        long totalRank = 0;
        long positiveNum = 0;
        
        for(int i = 0; i < classes.numInstances(); i++) {
            if(classes.instance(i).value(0) == 0) {
                totalRank += rankList.get(i);
                positiveNum++;
            }
        }
        double auc = (totalRank - (Math.pow(positiveNum, 2.0)+positiveNum) / 2) / (positiveNum * (classes.numInstances() - positiveNum));
        return auc;
    }
    
    
    /**
     * to store the file names contained in a folder
     * @param dataSetFilesPath the path of the folder
     */
    public static void buildDataSetsPathList(String dataSetFilesPath)
    {
        System.out.println(dataSetFilesPath);
        File filePath = new File(dataSetFilesPath);
        File DirProcessed = new File(dir_result + "\\Zero++");
        String[] fileNameList =  filePath.list();
        List<String> fileProcessed =  new ArrayList<String>();
        fileProcessed.addAll(Arrays.asList(DirProcessed.list()));
        int dataSetFileCount = 0;
        for (int count=0;count < fileNameList.length;count++)
        {
            if (fileNameList[count].toLowerCase().endsWith(".csv") || fileNameList[count].toLowerCase().endsWith(".arff"))
            {
            	// Verifica se o arquivo j� foi processado
            	if (!fileProcessed.contains(fileNameList[count]))
            		dataSetFileCount = dataSetFileCount +1;
            }
        }
        dataSetFullNameList = new String[dataSetFileCount];
        dataSetNameList = new String[dataSetFileCount];
        dataSetFileCount = 0;
        for (int count =0; count < fileNameList.length; count++)
        {
            if (fileNameList[count].toLowerCase().endsWith(".csv") || fileNameList[count].toLowerCase().endsWith(".arff"))
            {
            	if(!processado(fileNameList[count])) {
	                dataSetFullNameList[dataSetFileCount] = dataSetFilesPath+"\\"+fileNameList[count];
	                String[] parts = fileNameList[count].split("\\.");
	                String extension = parts[parts.length-1];
	                dataSetNameList[dataSetFileCount] = fileNameList[count].substring(0,fileNameList[count].lastIndexOf("." + extension));
	                //System.out.println(DataSetNameList[DataSetFileCount]);
	                dataSetFileCount = dataSetFileCount +1;
            	}
            }
        }
    }
    
    public static boolean processado(String dataset) {
    	File tmp = new File(dir_result + "\\Zero++\\" + dataset);
    	if (tmp.exists()) {
    		return true;
    	}
    	return false;
    }
    
    /**
     * to read data from a specific file
     * @param dataSetFileFullPath the full path of the file
     */
    public static Instances readDataSet(String dataSetFileFullPath)
    {
        Instances instances;
        try
        {
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(dataSetFileFullPath);
            instances =  source.getDataSet();
            instances.setClassIndex(instances.numAttributes() - 1);
        }
        catch (Exception e)
        {
            instances = null;
        }
        return instances;
    }
    
    /**
     * Splits the class attribute away. Depending on the invert flag, the instances without class attribute or only the class attribute of all instances is returned
     * @param instances the instances
     * @param invert flag; if true only the class attribute remains, otherwise the class attribute is the only attribute that is deleted.
     * @throws Exception exception if instances cannot be split
     * @return Instances without the class attribute or instances with only the class attribute
     */
    public static Instances divide(Instances instances, boolean invert) throws Exception{
        
        Instances newInstances = new Instances(instances);
        if(instances.classIndex() < 0)
            throw new Exception("A class attribute has to be specified.");
        if(invert){
            for(int i=0;i<newInstances.numAttributes();i++){
                if(i!=newInstances.classIndex()){
                    newInstances.deleteAttributeAt(i);
                    i--;
                }
            }
            return newInstances;
        }
        else{
            newInstances.setClassIndex(-1);
            newInstances.deleteAttributeAt(instances.classIndex());
            return newInstances;
        }
    }
    
    
    public static void outputSTDandMEAN(double d[],int run_num) {
        double mean = 0, std = 0;
        for(int i = 0; i < d.length; i++) {
            mean += d[i];
        }
        mean /= d.length;
        
        for(int i = 0; i < d.length; i++) {
            std += Math.pow(mean - d[i],2.0);
        }
        std = Math.sqrt(std/(d.length-1));
        System.out.println(formatOutput(mean)+","+formatOutput(2*std / Math.sqrt(d.length))
                +","+formatOutput(timeofBuilding / run_num /1000) + ","
                +formatOutput(timeofTesting / run_num /1000)+","
                +formatOutput((timeofBuilding+timeofTesting) / run_num /1000));
    }
    
    /**
     * to format the output value
     * @param outputValue the value intend to output
     * @return the formated value
     */
    public static String formatOutput(double outputValue)
    {
        DecimalFormat doubleFormat = new DecimalFormat("#.0000");
        return doubleFormat.format(outputValue);
    }
}
