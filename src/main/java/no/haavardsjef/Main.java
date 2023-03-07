package no.haavardsjef;


import no.haavardsjef.utility.DataLoader;

import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException {
        System.out.println("Hello world!");

        DataLoader dataLoader = new DataLoader();
        dataLoader.loadData();
    }
}