package org.simulator.math;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.factory.LinearSolverFactory_DDRM;
import org.ejml.interfaces.linsol.LinearSolverDense;
import org.ejml.simple.SimpleMatrix;
import org.junit.Test;
import org.simulator.math.MatrixOperations.MatrixException;

import java.util.Random;

import static org.junit.Assert.assertArrayEquals;

public class MatrixOperationsTest {
    @Test
    public void luSolverTest() throws MatrixException {
        // Make the test deterministic
        Random random = new Random(2718281828459045235L);

        for (int attempt = 0; attempt < 1000; attempt++) {
            final int n = 2 + random.nextInt(25);
            SimpleMatrix matrixA = SimpleMatrix.random_DDRM(n, n, -1e50, 1e50, random);
            double[][] a = new double[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    a[i][j] = matrixA.get(i, j);
                }
            }
            double[] b = SimpleMatrix.random_DDRM(1, n, -1e10, 1e10, random).getDDRM().data;

            // solve2 modifies its inputs, so run solve1 first
            double[] solution1 = solve1(a, b);
            double[] solution2 = solve2(a, b);
            assertArrayEquals(solution1, solution2, 1e-8d);
        }
    }

    private static double[] solve1(final double[][] a, final double[] b) {
        final int n = b.length;
        final DMatrixRMaj matrixB = new DMatrixRMaj(b);

        LinearSolverDense<DMatrixRMaj> lu = LinearSolverFactory_DDRM.lu(n);
        lu.setA(new DMatrixRMaj(a));
        lu.solve(matrixB, matrixB);
        return matrixB.data;
    }

    // Modifies both inputs
    private static double[] solve2(double[][] a, double[] b) throws MatrixException {
        final int n = b.length;
        int[] indx = new int[n];

        MatrixOperations.ludcmp(a, indx);
        MatrixOperations.lubksb(a, indx, b);
        return b;
    }
}
