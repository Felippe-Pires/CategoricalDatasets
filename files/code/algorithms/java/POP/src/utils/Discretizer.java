package utils;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

public class Discretizer {

    public static void main(String[] args) {
        try {
        	
        	String path = "C:\\Users\\pipip\\Google Drive\\Doutorado\\Pesquisa\\Experimentos\\Categorial_Data\\database\\Ready\\teste";	
            // Ler o arquivo CSV
        	ConverterUtils.DataSource source = new ConverterUtils.DataSource(path + "\\kddcup99.csv");
        	Instances data =  source.getDataSet();

            // Discretizar colunas numéricas
            Discretize discretizer = new Discretize();
            discretizer.setInputFormat(data);
            Instances newData = Filter.useFilter(data, discretizer);

            // Escrever o novo arquivo CSV
            try (FileWriter writer = new FileWriter(path + "\\arquivo_discretizado.csv")) {
                writer.write(newData.toString());
            }

            System.out.println("Arquivo discretizado salvo com sucesso!");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
