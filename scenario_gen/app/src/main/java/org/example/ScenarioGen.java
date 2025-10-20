package org.example;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.chocosolver.solver.Model;
import org.chocosolver.solver.Solver;
import org.chocosolver.solver.search.strategy.Search;
import org.chocosolver.solver.search.strategy.selectors.values.IntDomainMin;
import org.chocosolver.solver.search.strategy.selectors.variables.VariableSelector;
import org.chocosolver.solver.variables.BoolVar;
import org.chocosolver.solver.variables.IntVar;

import java.util.*;

/**
 * Scenario generator for a multi-agent Search-and-Rescue grid.
 * - Grid is H x W (default 20x20).
 * - Place N agents, M targets, K obstacles.
 * - No overlaps; agents/targets avoid obstacles.
 * - Each agent is at least MIN_MANHATTAN cells away from each target.
 * - Outputs JSON: {grid: {H,W}, agents:[{x,y}], targets:[{x,y}], obstacles:[{x,y}]}
 *
 * Choco 4.10.15 compatible (no SAT helpers).
 */
public class ScenarioGen {

    // ------------ Configuration ------------
    static class Config {
        int H = 20;            // grid height (y in [0..H-1])
        int W = 20;            // grid width  (x in [0..W-1])
        int N_AGENTS = 2;
        int N_TARGETS = 3;
        int N_OBS = 20;        // number of obstacle cells
        int MIN_MANHATTAN = 5; // min Manhattan distance agent <-> target
        Long SEED = (long) 1234;
        boolean requireAtLeastOneFreeRingAroundAgents = true; // optional safety
    }

    static class XY {
        int x, y;
        XY(int x, int y) { this.x = x; this.y = y; }
    }

    static class ScenarioOut {
        Map<String, Integer> grid = new LinkedHashMap<>();
        List<XY> agents = new ArrayList<>();
        List<XY> targets = new ArrayList<>();
        List<XY> obstacles = new ArrayList<>();
    }

    public static void main(String[] args) {
        Config cfg = parseArgs(args);

        Model model = new Model("SAR-Scenario");

        // --- Variables ---
        IntVar[] ax = new IntVar[cfg.N_AGENTS];
        IntVar[] ay = new IntVar[cfg.N_AGENTS];
        for (int i = 0; i < cfg.N_AGENTS; i++) {
            ax[i] = model.intVar("ax_" + i, 0, cfg.W - 1);
            ay[i] = model.intVar("ay_" + i, 0, cfg.H - 1);
        }

        IntVar[] tx = new IntVar[cfg.N_TARGETS];
        IntVar[] ty = new IntVar[cfg.N_TARGETS];
        for (int t = 0; t < cfg.N_TARGETS; t++) {
            tx[t] = model.intVar("tx_" + t, 0, cfg.W - 1);
            ty[t] = model.intVar("ty_" + t, 0, cfg.H - 1);
        }

        IntVar[] ox = new IntVar[cfg.N_OBS];
        IntVar[] oy = new IntVar[cfg.N_OBS];
        for (int k = 0; k < cfg.N_OBS; k++) {
            ox[k] = model.intVar("ox_" + k, 0, cfg.W - 1);
            oy[k] = model.intVar("oy_" + k, 0, cfg.H - 1);
        }

        // --- Constraints ---

        // 1) Distinct agents
        for (int i = 0; i < cfg.N_AGENTS; i++) {
            for (int j = i + 1; j < cfg.N_AGENTS; j++) {
                model.or(
                        model.arithm(ax[i], "!=", ax[j]),
                        model.arithm(ay[i], "!=", ay[j])
                ).post();
            }
        }

        // 2) Distinct targets
        for (int i = 0; i < cfg.N_TARGETS; i++) {
            for (int j = i + 1; j < cfg.N_TARGETS; j++) {
                model.or(
                        model.arithm(tx[i], "!=", tx[j]),
                        model.arithm(ty[i], "!=", ty[j])
                ).post();
            }
        }

        // 3) Distinct obstacles
        for (int i = 0; i < cfg.N_OBS; i++) {
            for (int j = i + 1; j < cfg.N_OBS; j++) {
                model.or(
                        model.arithm(ox[i], "!=", ox[j]),
                        model.arithm(oy[i], "!=", oy[j])
                ).post();
            }
        }

        // 4) Agents/Targets must avoid obstacles
        for (int i = 0; i < cfg.N_AGENTS; i++) {
            for (int k = 0; k < cfg.N_OBS; k++) {
                model.or(
                        model.arithm(ax[i], "!=", ox[k]),
                        model.arithm(ay[i], "!=", oy[k])
                ).post();
            }
        }
        for (int t = 0; t < cfg.N_TARGETS; t++) {
            for (int k = 0; k < cfg.N_OBS; k++) {
                model.or(
                        model.arithm(tx[t], "!=", ox[k]),
                        model.arithm(ty[t], "!=", oy[k])
                ).post();
            }
        }

        // 5) Agent–Target min Manhattan distance >= MIN_MANHATTAN
        for (int i = 0; i < cfg.N_AGENTS; i++) {
            for (int t = 0; t < cfg.N_TARGETS; t++) {
                IntVar dx = model.intVar("dx_a" + i + "_t" + t, 0, cfg.W - 1);
                IntVar dy = model.intVar("dy_a" + i + "_t" + t, 0, cfg.H - 1);
                model.distance(ax[i], tx[t], "=", dx).post(); // |ax - tx| = dx
                model.distance(ay[i], ty[t], "=", dy).post(); // |ay - ty| = dy
                model.sum(new IntVar[]{dx, dy}, ">=", cfg.MIN_MANHATTAN).post();
            }
        }

        // (Optional) 6) A safety ring: each agent has at least one free neighbor cell.
        if (cfg.requireAtLeastOneFreeRingAroundAgents) {
            // Build a fast obstacle membership test via reification:
            // For each cell (x,y), define BoolVar isObs_xy that is true iff some obstacle occupies it.
            BoolVar[][] isObs = new BoolVar[cfg.H][cfg.W];
            for (int y = 0; y < cfg.H; y++) {
                for (int x = 0; x < cfg.W; x++) {
                    BoolVar cell = model.boolVar("isObs_" + x + "_" + y);
                    // cell <-> OR_k ((ox[k]==x) AND (oy[k]==y))
                    BoolVar[] hits = new BoolVar[cfg.N_OBS];
                    for (int k = 0; k < cfg.N_OBS; k++) {
                        BoolVar hit = model.boolVar();
                        model.and(
                                model.arithm(ox[k], "=", x),
                                model.arithm(oy[k], "=", y)
                        ).reifyWith(hit);
                        hits[k] = hit;
                    }

                    IntVar sumHits = model.intVar(0, cfg.N_OBS);
                    model.sum(hits, "=", sumHits).post();
                    // cell = 1  -> sumHits >= 1
                    model.ifThen(
                            model.arithm(cell, "=", 1),
                            model.arithm(sumHits, ">=", 1)
                    );
                    // sumHits >= 1 -> cell = 1
                    model.ifThen(
                            model.arithm(sumHits, ">=", 1),
                            model.arithm(cell, "=", 1)
                    );
                    isObs[y][x] = cell;
                }
            }
            // For each agent, at least one 4-neighbor must be free (not obstacle).
            int[][] dirs = {{1,0},{-1,0},{0,1},{0,-1}};
            for (int i = 0; i < cfg.N_AGENTS; i++) {
                // Build disjunction over neighbors being free
                // Reify "agent at (x,y)" to check neighbor cells. Do it by scanning all cells:
                BoolVar hasFreeNeighbor = model.boolVar("agent_" + i + "_hasFreeNbr");
             
                List<BoolVar> lits = new ArrayList<>();
                for (int y = 0; y < cfg.H; y++) {
                    for (int x = 0; x < cfg.W; x++) {
                        // Build OR of free neighbor:
                        List<BoolVar> freeNbrs = new ArrayList<>();
                        for (int[] d : dirs) {
                            int nx = x + d[0], ny = y + d[1];
                            if (0 <= nx && nx < cfg.W && 0 <= ny && ny < cfg.H) {
                                BoolVar free = model.boolVar();
                                // free <-> isObs[ny][nx] == 0
                                model.arithm(isObs[ny][nx], "=", 0).reifyWith(free);
                                freeNbrs.add(free);
                            }
                        }
                        if (freeNbrs.isEmpty()) continue;
                        BoolVar orFree = model.boolVar();
                   
                        IntVar sumFree = model.intVar(0, freeNbrs.size());
                        model.sum(freeNbrs.toArray(new BoolVar[0]), "=", sumFree).post();
                        model.ifThen(
                                model.arithm(sumFree, ">=", 1),
                                model.arithm(orFree, "=", 1)
                        );
                        model.ifThen(
                                model.arithm(orFree, "=", 1),
                                model.arithm(sumFree, ">=", 1)
                        );

                        BoolVar atXY = model.boolVar();
                        model.and(
                                model.arithm(ax[i], "=", x),
                                model.arithm(ay[i], "=", y)
                        ).reifyWith(atXY);

                        BoolVar lit = model.boolVar();
                        model.and(atXY, orFree).reifyWith(lit);
                        lits.add(lit);
                    }
                }

                if (!lits.isEmpty()) {
                    IntVar sumLits = model.intVar(0, lits.size());
                    model.sum(lits.toArray(new BoolVar[0]), "=", sumLits).post();
                    model.ifThen(
                            model.arithm(sumLits, ">=", 1),
                            model.arithm(hasFreeNeighbor, "=", 1)
                    );
                    model.ifThen(
                            model.arithm(hasFreeNeighbor, "=", 1),
                            model.arithm(sumLits, ">=", 1)
                    );

                    model.arithm(hasFreeNeighbor, "=", 1).post();
                }
            }
        }
        

        // --- Search strategy ---
        // Collect all IntVars into one array for a single search
        List<IntVar> all = new ArrayList<>();
        all.addAll(Arrays.asList(ax));
        all.addAll(Arrays.asList(ay));
        all.addAll(Arrays.asList(tx));
        all.addAll(Arrays.asList(ty));
        all.addAll(Arrays.asList(ox));
        all.addAll(Arrays.asList(oy));
        IntVar[] allVars = all.toArray(new IntVar[0]);

        VariableSelector<IntVar> varSel = (vars) -> {
            IntVar best = null;
            int bestSize = Integer.MAX_VALUE;
            for (IntVar v : vars) {
                if (v.isInstantiated()) continue;
                int ds = v.getDomainSize();
                if (ds < bestSize) {
                    best = v;
                    bestSize = ds;
                    if (bestSize == 1) break;
                } else if (ds == bestSize && best != null) {
                    if (v.getName().compareTo(best.getName()) < 0) best = v;
                }
            }
            return best;
        };
        model.getSolver().setSearch(Search.intVarSearch(varSel, new IntDomainMin(), allVars));

        // Random seed for reproducibility
        if (cfg.SEED != null) {
            model.getSolver().setSearch(Search.randomSearch(allVars, cfg.SEED));
        }

        // --- Solve ---
        Solver solver = model.getSolver();
        boolean ok = solver.solve();
        System.out.println("Solved: " + ok);
        System.out.println("Solutions: " + solver.getSolutionCount());
        System.out.println("Nodes: " + solver.getNodeCount());
        System.out.println("Fails: " + solver.getFailCount());
        System.out.println("Time (ms): " + solver.getTimeCount());


        if (!ok) {
            System.err.println("No scenario found with current constraints.");
            System.exit(2);
        }

        // --- Collect solution & print JSON ---
        ScenarioOut out = new ScenarioOut();
        out.grid.put("H", cfg.H);
        out.grid.put("W", cfg.W);

        for (int i = 0; i < cfg.N_AGENTS; i++) {
            out.agents.add(new XY(ax[i].getValue(), ay[i].getValue()));
        }
        for (int t = 0; t < cfg.N_TARGETS; t++) {
            out.targets.add(new XY(tx[t].getValue(), ty[t].getValue()));
        }
    
        Set<Long> seen = new HashSet<>();
        for (int k = 0; k < cfg.N_OBS; k++) {
            int x = ox[k].getValue(), y = oy[k].getValue();
            long key = (((long) y) << 32) ^ (x & 0xffffffffL);
            if (seen.add(key)) out.obstacles.add(new XY(x, y));
        }

        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        System.out.println(gson.toJson(out));
    }

    // ------------ CLI parsing ------------
    private static Config parseArgs(String[] args) {
        Config c = new Config();
       
        for (String a : args) {
            if (a.startsWith("--H=")) c.H = Integer.parseInt(a.substring(4));
            else if (a.startsWith("--W=")) c.W = Integer.parseInt(a.substring(4));
            else if (a.startsWith("--agents=")) c.N_AGENTS = Integer.parseInt(a.substring(9));
            else if (a.startsWith("--targets=")) c.N_TARGETS = Integer.parseInt(a.substring(10));
            else if (a.startsWith("--obs=")) c.N_OBS = Integer.parseInt(a.substring(6));
            else if (a.startsWith("--minMan=")) c.MIN_MANHATTAN = Integer.parseInt(a.substring(9));
            else if (a.startsWith("--seed=")) c.SEED = Long.parseLong(a.substring(7));
            else if (a.startsWith("--ring=")) c.requireAtLeastOneFreeRingAroundAgents = Integer.parseInt(a.substring(7)) != 0;
        }
        return c;
    }
}
