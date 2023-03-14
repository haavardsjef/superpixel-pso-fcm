package no.haavardsjef.classification;

import java.util.List;

public interface IClassifier {

    public void evaluate(List<Integer> selectedBands); // TODO: Make this return some kind of result object
}
