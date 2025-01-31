#include <iostream>
#include <fstream>
#include <set>
#include <unordered_set>
#include <limits>
#include <sstream>
#include <iomanip>
#include <set>

#include "nlohmann/json.hpp"

using namespace std;
using json = nlohmann::json;

// begin trivial helper stuff
ostream& dbg = cerr;

void fail (const string &s) {
    cout << "FAIL: " << s << endl;
    dbg << "FAIL: " << s << endl;
    exit(1);
}

void warn (const string &s) {
    dbg << "WARNING: " << s << endl;
}

#define DBG(vari) cerr<<"["<<__LINE__<<"] "<<#vari<<" = "<<(vari)<<endl;

template <typename T>
ostream& operator << (ostream &s, const vector<T> &v) {
    for (const T &x : v) {
        s << x << " ";
    }
    return s;
}

template <typename T>
string to_string (const vector<T> &v) {
    stringstream ss;
    ss << v;
    return ss.str();
}

template <typename T>
void append (vector<T> &v, const vector<T> &w) {
    v.insert(v.end(), w.begin(), w.end());
}

template <typename T>
inline void minify (T &x, const T &y) {
    x = min(x,y);
}

int ceildiv (int x, int y) {
    assert(y > 0);
    return (x + y - 1) / y;
}

constexpr double INFTY = 1e30;

vector<int> vectorOfSetBits (const vector<bool> &v) {
    vector<int> res;
    for (int i = 0; i < v.size(); ++i) {
        if (v[i]) {
            res.push_back(i);
        }
    }
    return res;
}

// end trivial helper stuff


constexpr int DOWNSETS_LIMIT = 40'000;
constexpr int DOWNSETS_EXPLORATION_LIMIT = 200'000;
constexpr int DEVICES_LIMIT = 10'000; // some loose upper bound on number of devices there can be in any reasonable input
constexpr bool DATA_PARALLEL_DEGREE_MUST_DIVIDE_BATCH_SIZE = false;
constexpr bool DATA_PARALLEL_DEGREE_MUST_DIVIDE_NUM_DEVICES = false;

struct Node {
    // Node represents a layer,
    // in a graph where the TMPC width t is *already fixed*
    int id; // v
    double parameterSize; // size of weights
    double activationSize; // sum of ALL (also intermediate) activation sizes
    double optimalLatencyFw; // computed in the per-layer optimization problem (ILP etc.)
    double optimalLatencyBw;
    bool isTensorParallelized; // if YES, the node represents a layer *slice*
};

void from_json (const json &j, Node &n) {
    j.at("id").get_to(n.id);
    j.at("parameterSize").get_to(n.parameterSize);
    j.at("activationSize").get_to(n.activationSize);
    j.at("optimalLatencyFw").get_to(n.optimalLatencyFw);
    j.at("optimalLatencyBw").get_to(n.optimalLatencyBw);
    j.at("isTensorParallelized").get_to(n.isTensorParallelized);
}

struct Edge {
    int sourceId; // u
    int destId; // v
    double communicationCost; // c(u,v), in bytes
};

void from_json (const json &j, Edge &e) {
    j.at("sourceId").get_to(e.sourceId);
    j.at("destId").get_to(e.destId);
    j.at("communicationCost").get_to(e.communicationCost);
}

struct Instance {
    double maxMemoryPerDevice;
    int maxDevices;
    double bandwidth;
    int mbsInBatch;
    map<int, vector<Node>> nodes; // nodes[t] are for graph of TMPC width t
    map<int, vector<Edge>> edges; // edges[t] are for graph of TMPC width t
    bool activationRecomputation;
    string optimizerAlgorithm; // SGD or Adam (in the latter case the memory usage becomes 3* rather than 2*parameterSize)
    int fixedDPStrategy[3];
    int numTransformerLayers;

    // filled with renumber()
    unordered_map<int,int> newNumber;
    vector<int> oldNumber;

    void checkInputCorrectness() const;
    vector<int> isDAG() const;
    void renumber();    
};

void from_json (const json &j, Instance &ii) {
    j.at("maxMemoryPerDevice").get_to(ii.maxMemoryPerDevice);
    j.at("maxDevices").get_to(ii.maxDevices);
    j.at("bandwidth").get_to(ii.bandwidth);
    j.at("mbsInBatch").get_to(ii.mbsInBatch);
    j.at("fixedPP").get_to(ii.fixedDPStrategy[0]);
    j.at("fixedTMPC").get_to(ii.fixedDPStrategy[1]);
    j.at("fixedDP").get_to(ii.fixedDPStrategy[2]);
    j.at("numTransformerLayers").get_to(ii.numTransformerLayers);
    map<string, vector<Node>> nodes;
    j.at("nodes").get_to(nodes);
    for (const auto &p : nodes) {
        ii.nodes[stoi(p.first)] = p.second;
    }
    map<string, vector<Edge>> edges;
    j.at("edges").get_to(edges);
    for (const auto &p : edges) {
        ii.edges[stoi(p.first)] = p.second;
    }
    j.at("activationRecomputation").get_to(ii.activationRecomputation);
    j.at("optimizerAlgorithm").get_to(ii.optimizerAlgorithm);

    ii.checkInputCorrectness();
    ii.renumber();
    ii.checkInputCorrectness();
}

void Instance::checkInputCorrectness() const {
    if (maxDevices < 1 || maxDevices > DEVICES_LIMIT) {
        fail("wrong number of devices");
    }
    if (bandwidth < 1e-9) {
        fail("wrong bandwidth");
    }
    if (maxMemoryPerDevice < 1e-9) {
        fail("wrong maxMemoryPerDevice");
    }
    if (mbsInBatch < 1) {
        fail("wrong mbsInBatch");
    }
    if (optimizerAlgorithm != "SGD" && optimizerAlgorithm != "Adam") {
        fail("optimizerAlgorithm should be SGD or Adam");
    }
    if (nodes.empty()) {
        fail("no graphs/TMPCwidths in input");
    }
    set<int> tmpcWidths;
    for (const auto &p : nodes) {
        tmpcWidths.insert(p.first);
        if (p.first > DEVICES_LIMIT) {
            fail("TMPC width clearly too large");
        }
        if (p.first < 1) {
            fail("TMPC width < 1?");
        }
    }
    set<int> edgeTmpcWidths;
    for (const auto &p : edges) {
        edgeTmpcWidths.insert(p.first);
    }
    if (tmpcWidths != edgeTmpcWidths) {
        fail("graphs and edges have different TMPC width sets");
    }
    set<int> nodeIds;
    for (const Node &n : nodes.begin()->second) {
        nodeIds.insert(n.id);
    }
    if (nodeIds.empty()) {
        fail("no nodes in graph");
    }
    if (nodeIds.size() != nodes.begin()->second.size()) {
        fail("node ids are not unique");
    }
    for (const auto &p : nodes) {
        set<int> nodeIdsThisTmpc;
        for (const Node &n : p.second) {
            if (n.parameterSize < 0) {
                fail("parameterSize < 0");
            }
            if (n.activationSize < 0) {
                fail("activationSize < 0");
            }
            if (n.optimalLatencyFw < 0) {
                fail("optimalLatencyFw < 0");
            }
            if (n.optimalLatencyBw < 0) {
                fail("optimalLatencyBw < 0");
            }
            if (p.first == 1 && n.isTensorParallelized) {
                fail("TMPC width 1 but node isTensorParallelized");
            }
            nodeIdsThisTmpc.insert(n.id);
        }
        if (nodeIdsThisTmpc != nodeIds) {
            fail("node (layer) ids are not the same for all TMPC widths");
        }
    }
    set<pair<int,int>> edgeIds;
    for (const Edge &e : edges.begin()->second) {
        edgeIds.insert({e.sourceId, e.destId});
    }
    if (edgeIds.size() != edges.begin()->second.size()) {
        fail("parallel edges exist (maybe not a problem but let's rather contract them)");
    }
    for (const auto &p : edges) {
        set<pair<int,int>> edgeIdsThisTmpc;
        for (const Edge &e : p.second) {
            if (nodeIds.count(e.sourceId) == 0) {
                fail("edge sourceId not in nodeIds");
            }
            if (nodeIds.count(e.destId) == 0) {
                fail("edge destId not in nodeIds");
            }
            if (e.sourceId == e.destId) {
                fail("edge sourceId == destId (self-loop)");
            }
            if (e.communicationCost < 0) {
                fail("communicationCost < 0");
            }
            bool inserted = edgeIdsThisTmpc.insert({e.sourceId, e.destId}).second;
            if (inserted == false) {
                fail("parallel edges exist (maybe not a problem but let's rather contract them)");
            }
        }
        if (edgeIds != edgeIdsThisTmpc) {
            fail("edges are not the same for all TMPC widths");
        }
    }
    if (isDAG().empty()) {
        fail("graph is not a DAG");
    }
}


// returns empty vector if not a DAG, otherwise topological order
vector<int> Instance::isDAG() const {
    unordered_map<int,int> indegree;
    unordered_map<int,vector<int>> outgoingEdges;
    for (const Edge &e : edges.begin()->second) {
        ++indegree[e.destId];
        outgoingEdges[e.sourceId].push_back(e.destId);
    }
    vector<int> deg0vertices;
    for (const Node &n : nodes.begin()->second) {
        if (indegree[n.id] == 0) {
            deg0vertices.push_back(n.id);
        }
    }
    vector<int> verticesInTopologicalOrder;
    while (!deg0vertices.empty()) {
        int v = deg0vertices.back();
        deg0vertices.pop_back();
        verticesInTopologicalOrder.push_back(v);
        for (int w : outgoingEdges[v]) {
            --indegree[w];
            if (indegree[w] == 0) {
                deg0vertices.push_back(w);
            }
        }
    }
    if (verticesInTopologicalOrder.size() != nodes.begin()->second.size()) {
        return vector<int>();
    } else {
        return verticesInTopologicalOrder;
    }
}


void Instance::renumber () {
    // renumber nodes as 0,1,2,... in a topological order
    assert(oldNumber.empty());
    // build oldNumber and newNumber
    oldNumber = isDAG();
    if (oldNumber.empty()) {
        fail("graph is not a DAG");
    }
    for (int i = 0; i < oldNumber.size(); ++i) {
        newNumber[oldNumber[i]] = i;
    }
    // now replace old ids with new ids everywhere
    for (auto &it : nodes) {
        for (Node &n : it.second) {
            n.id = newNumber[n.id];
        }
    }
    for (auto &it : edges) {
        for (Edge &e : it.second) {
            e.sourceId = newNumber[e.sourceId];
            e.destId = newNumber[e.destId];
        }
    }
}

struct ResultStage {
    vector<int> nodes; // (new ids, unless renumberResultBack was run)
    int devicesForStage;
};

void to_json(json &j, const ResultStage &s) {
    j = json{{"nodes", s.nodes},
             {"devicesForStage", s.devicesForStage}
    };
}

struct Result {
    vector<ResultStage> stages;
    // string debugInfo;
    int dataParallelDegree;
    int tensorParallelDegree;
};

void to_json(json &j, const Result &r) {
    j = json{{"stages", r.stages},
             {"dataParallelDegree", r.dataParallelDegree},
             {"tensorParallelDegree", r.tensorParallelDegree}
    };
}


struct LoadOfStage {
    double fw_bw_latency_with_recompute; // max_device (fw + (fw + bw))
    double fw_bw_latency_wo_recompute; // max_device (fw + (just bw))
    double parameter_size; // max_device (parameter_size)
    int max_s_memory_feasible; // max s for which each device has enough memory
};


struct Graph {
    // this is already for a fixed TMPC width t!
    const int tmpcWidth;
    const Instance &ins; // already renumbered (nodes 0,1,2,...)
    const int boundOnS; // set as min(maxDevices, # nodes)

    vector<vector<pair<int,double>>> incomingEdges; // v -> vector of {u, c(u,v)}
    vector<vector<pair<int,double>>> outgoingEdges; // v -> vector of {w, c(v,w)}
    vector<const Node*> node; // node[v] = pointer to node with (new) id v

    // downsets, represented as indicator vectors
    unordered_map<vector<bool>,int> downsetToId; // map downset to its ID
    vector<vector<bool>> downsets; // maps ID to downset
    vector<int> downsetsSortedBySize; // IDs of downsets, sorted by size

    // pairs of downsets (that induce contiguous sets)
    vector<vector<int>> immediateSubDownsets;
    // immediateSubDownsets[id] = IDs of downsets that are immediate subsets of the downset with ID id
    vector<vector<int>> subDownsets;
    // subDownsets[id] = IDs of downsets that are subsets of the downset with ID id
    // (this takes O(numberOfDownsetPairs) space; could be done on the fly in the DP perhaps,
    //  but if one can't afford this memory then probably one can't afford the DP alg timewise)
    long long numberOfDownsetPairs;

    Graph (const Instance &_ins, int _tmpcWidth);
    void generateDownsets();
    void growDownset(const vector<bool> &downset, int myId);
    void prepareSubDownsets();

    double getDataParallelResyncCost (int d, double parameterSize) const;

    vector<bool> getContiguousSet (int id, int subId) const;
    
    // a -> loadOfStage(downsets[id] \ downsets[subId], a many devices)
    // (only some a will appear, namely those that yield smaller load than a-1)
    // (e.g. if there is no branching in the layer graph, then only a=1 or a=t can make sense)
    map<int,LoadOfStage> getLoadOfStage (int id, int subId) const; // wrapper
    map<int,LoadOfStage> getLoadOfStage (const vector<int> &nodes) const; // wrapper, used in reconstruction
    map<int,LoadOfStage> getLoadOfStage (const vector<int> &nodes, const vector<bool> &nodesVB) const;
    void getLoadOfStageForA (const vector<int> &nodes, const vector<bool> &nodesVB,
                             int a, map<int,LoadOfStage> &resultMap) const;

    vector<vector<vector<double>>> dp; // dp table
    // dp[downset id][num devices][num stages] -> minimal max-load of a device
    int idOfFullSet;
    Result runDP();

    void renumberResultBack (Result &r) const;

    double getTimePerBatchForResult (const Result &r) const;
};


Graph::Graph (const Instance &_ins, int _tmpcWidth)
  : tmpcWidth(_tmpcWidth),
    ins(_ins),
    boundOnS(min(ins.maxDevices, (int)ins.nodes.at(tmpcWidth).size())),
    numberOfDownsetPairs(0)
{
    // build incomingEdges, outgoingEdges, node
    incomingEdges.resize(ins.nodes.at(tmpcWidth).size());
    outgoingEdges.resize(ins.nodes.at(tmpcWidth).size());
    for (const Edge &e : ins.edges.at(tmpcWidth)) {
        incomingEdges[e.destId].push_back({e.sourceId, e.communicationCost});
        outgoingEdges[e.sourceId].push_back({e.destId, e.communicationCost});
    }
    node.resize(ins.nodes.at(tmpcWidth).size());
    for (const Node &n : ins.nodes.at(tmpcWidth)) {
        node[n.id] = &n;
    }
    // generate downsets
    generateDownsets();
    // immediateSubDownsets is prepared. now prepare subDownsets
    prepareSubDownsets();
}


void Graph::generateDownsets () {
    if (!downsets.empty()) {
        fail("downsets not empty. generating downsets twice?");
    }

    // start with empty set
    const vector<bool> emptySet(ins.nodes.at(tmpcWidth).size(), false);
    downsetToId[emptySet] = 0;
    downsets.push_back(emptySet);
    immediateSubDownsets.emplace_back();
    growDownset(emptySet, 0);

    dbg << "generated " << downsets.size() << " downsets" << endl;
    if (downsets.size() > DOWNSETS_LIMIT) {
        fail("too many downsets (current limit set at " + to_string(DOWNSETS_LIMIT) + "); this isn't going to work...");
    }

    idOfFullSet = downsetToId.at(vector<bool>(ins.nodes.at(tmpcWidth).size(), true));

    // prepare downsetsSortedBySize
    vector<pair<int,int>> sorter; // {<size, downset id>}
    for (int i = 0; i < downsets.size(); ++i) {
        sorter.emplace_back(count(downsets[i].begin(), downsets[i].end(), true), i);
    }
    sort(sorter.begin(), sorter.end());
    for (auto &p : sorter) {
        downsetsSortedBySize.push_back(p.second);
    }
    assert(downsetsSortedBySize[0] == 0);
}


void Graph::growDownset (const vector<bool> &downset, int myId) {
    // myId == downsetToId[downset]
    // try to add every vertex
    for (int v = 0; v < node.size(); ++v) {
        if (!downset[v]) {
            // try downset + {v} as a new downset
            // check if valid: do all v's successors belong to downset?
            bool valid = true;
            for (const pair<int,double> &p : outgoingEdges[v]) {
                // edge v -> p.first
                if (!downset[p.first]) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                vector<bool> newDownset = downset;
                newDownset[v] = true;
                // check if newDownset had already been generated
                if (!downsetToId.count(newDownset)) {
                    // new downset
                    int newId = downsets.size();
                    downsetToId[newDownset] = newId;
                    downsets.push_back(newDownset);
                    if (downsets.size() >= DOWNSETS_EXPLORATION_LIMIT) {
                        fail("already over " + to_string(DOWNSETS_EXPLORATION_LIMIT) + " downsets. this isn't going to work...");
                    }
                    immediateSubDownsets.emplace_back();
                    growDownset(newDownset, newId);
                }
                immediateSubDownsets[downsetToId[newDownset]].push_back(myId);
            }
        }
    }
}


void Graph::prepareSubDownsets () {
    // subDownsets = transitive closure of immediateSubDownsets

    if (numberOfDownsetPairs != 0) {
        fail("prepareSubDownsets() called twice?");
    }
    subDownsets.resize(downsets.size());

    for (int id = 0; id < downsets.size(); ++id) {
        // we will generate subIdeals[id] using some BFS/DFS
        vector<int> queue = {id};
        unordered_set<int> enqueuedDownsets = {id};
        while (!queue.empty()) {
            int subId = queue.back();
            queue.pop_back();

            // now visiting subId
            if (subId != id) {
                subDownsets[id].push_back(subId);
                ++numberOfDownsetPairs;
            }

            // expand further from subId
            for (int subSubId : immediateSubDownsets[subId]) {
                if (enqueuedDownsets.insert(subSubId).second == true) {
                    // subSubId was not in enqueuedIdeals before
                    queue.push_back(subSubId);
                }
            }
        }
    }

    dbg << "numberOfDownsetPairs = " << numberOfDownsetPairs << endl;
}


// returns the difference downsets[id] \ downsets[subId] as vector<bool>
vector<bool> Graph::getContiguousSet (int id, int subId) const {
    vector<bool> downset = downsets[id], subDownset = downsets[subId];
    for (int v = 0; v < ins.nodes.at(tmpcWidth).size(); ++v) {
        if (subDownset[v]) {
            downset[v] = false;
        }
    }
    return downset;
}


double Graph::getDataParallelResyncCost (int d, double parameterSize) const {
    return 4.0 * (d-1) * parameterSize / ins.bandwidth / d;
}


Result Graph::runDP () {
    // initialize DP table: dp[downset][k][s]
    // (partition downset over AT MOST k devices/accelerators, with EXACTLY s stages)
    dp.assign(downsets.size(), vector<vector<double>>(
              ins.maxDevices+1, vector<double>(
              boundOnS+1, INFTY)));

    // case of the empty set (downset with ID 0)
    for (int k = 0; k <= ins.maxDevices; ++k) {
        dp[0][k][0] = 0;
    }
    // dp[][][] will remain forever monotone w.r.t. k

    // profiling stuff
    double timeSpentInGetLoadOfStage = 0.0;

    // here we go!
    dbg << "running DP..." << endl;
    const clock_t startTimeDP = clock();
    for (int id : downsetsSortedBySize) {
        if (id == 0) continue; // already filled above

        if (id == idOfFullSet) break; // will be handled separately below

        // we want to fill dp[id][*][*] (already initialized to INFTY).
        // we will loop over every subdownset subId (their list is already
        // precomputed in subDownsets[id] for convenience)
        for (int subId : subDownsets[id]) {

            // putting downsets[id] \ downsets[subId] (contiguous set) as the next stage

            const clock_t startTimeGetLoadOfStage = clock();
            map<int,LoadOfStage> loadOfStage = getLoadOfStage(id, subId);
            timeSpentInGetLoadOfStage += (clock() - startTimeGetLoadOfStage) * 1.0 / CLOCKS_PER_SEC;

            for (const pair<int,LoadOfStage> &it : loadOfStage) { // loop over a
                const int a = it.first;
                const int max_s = min(it.second.max_s_memory_feasible, boundOnS);
                for (int s = 1; s <= max_s; ++s) {
                    const double load = (ins.activationRecomputation && s > 1)
                            ? it.second.fw_bw_latency_with_recompute
                            : it.second.fw_bw_latency_wo_recompute;
#pragma GCC unroll 16
                    for (int k = a; k <= ins.maxDevices; ++k) {
                        minify(dp[id][k][s], max(dp[subId][k-a][s-1], load));
                    }
                }
            }
        }
    }

    const double timeSpentInDPLoop = (clock() - startTimeDP) * 1.0 / CLOCKS_PER_SEC - timeSpentInGetLoadOfStage;
    // this is EXCLUDING the time spent in getLoadOfStage() calls
    const double timeSpentInGetLoadOfStage_mainDP = timeSpentInGetLoadOfStage;

    // final DP round (placing the first stage)
    const clock_t startTimeFinalDPRound = clock();
    double finalTimePerBatch = INFTY;
    int finalD = -1, finalS = -1, finalSubId = -1, finalA = -1;
    for (int d = 1; d <= ins.maxDevices && d <= ins.mbsInBatch; ++d) {
        // d = data-parallelism degree
        if (DATA_PARALLEL_DEGREE_MUST_DIVIDE_BATCH_SIZE && ins.mbsInBatch % d != 0) {
            continue;
        }
        if (DATA_PARALLEL_DEGREE_MUST_DIVIDE_NUM_DEVICES && ins.maxDevices % d != 0) {
            continue;
        }
        const int numDevicesPerPipeline = ins.maxDevices / d;
        const int mbsInBatchPerPipeline = ceildiv(ins.mbsInBatch, d);

        for (int subId : subDownsets[idOfFullSet]) {
            // putting (full set) \ downsets[subId] (contiguous set) as the first stage
            clock_t startTime = clock();
            map<int,LoadOfStage> loadOfStage = getLoadOfStage(idOfFullSet, subId);
            timeSpentInGetLoadOfStage += (clock() - startTime) * 1.0 / CLOCKS_PER_SEC;

            for (const pair<int,LoadOfStage> &it : loadOfStage) { // loop over a
                const int a = it.first;
                if (numDevicesPerPipeline < a) continue; // not enough devices
                const int max_s = min(it.second.max_s_memory_feasible, boundOnS);

                for (int s = 1; s <= max_s; ++s) {
                    // since this is the first stage, s = pipeline depth (number of stages)
                    if (dp[subId][numDevicesPerPipeline - a][s-1] > INFTY / 2) continue;
                    const double load = (ins.activationRecomputation && s > 1)
                            ? it.second.fw_bw_latency_with_recompute
                            : it.second.fw_bw_latency_wo_recompute;
                    const double timePerBatch =
                            max(dp[subId][numDevicesPerPipeline - a][s-1], load)
                            *
                            (mbsInBatchPerPipeline + s - 1)
                            +
                            getDataParallelResyncCost(d, it.second.parameter_size);
                    if (timePerBatch < finalTimePerBatch) {
                        finalTimePerBatch = timePerBatch;
                        finalD = d;
                        finalS = s;
                        finalA = a;
                        finalSubId = subId;
                    }
                }
            }
        }
    }
    const double timeSpentInFinalDPRound = (clock() - startTimeFinalDPRound) * 1.0 / CLOCKS_PER_SEC
                                    - (timeSpentInGetLoadOfStage - timeSpentInGetLoadOfStage_mainDP);
    // this is EXCLUDING the time spent in getLoadOfStage() calls

    if (finalTimePerBatch > INFTY/2) {
        // we didn't find a feasible solution
        dbg << "no feasible solution found" << endl;
        return Result();
    }
    dbg << "finalTimePerBatch = " << finalTimePerBatch << endl;

    // now we reconstruct the solution
    Result result;
    result.dataParallelDegree = finalD;
    result.tensorParallelDegree = tmpcWidth;
    // begin by placing the first stage
    ResultStage firstStage;
    firstStage.devicesForStage = finalA;
    firstStage.nodes = vectorOfSetBits(getContiguousSet(idOfFullSet, finalSubId));
    result.stages.push_back(firstStage);
    // now reconstruct the rest of the stages
    int curId = finalSubId, curS = finalS - 1, curK = ins.maxDevices / finalD - finalA;
    while (curId != 0) { // curId is not empty set
        assert(curK > 0);
        assert(curS > 0);
        // how does dp[curId][curK][curS] arise?
        bool found = false;
        for (int subId : subDownsets[curId]) {
            const clock_t startTimeGetLoadOfStage = clock();
            map<int,LoadOfStage> loadOfStage = getLoadOfStage(curId, subId);
            timeSpentInGetLoadOfStage += (clock() - startTimeGetLoadOfStage) * 1.0 / CLOCKS_PER_SEC;

            for (const pair<int,LoadOfStage> &it : loadOfStage) { // loop over a
                const int a = it.first;
                if (curK < a) continue; // not enough devices
                const double load = (ins.activationRecomputation && curS > 1)
                        ? it.second.fw_bw_latency_with_recompute
                        : it.second.fw_bw_latency_wo_recompute;
                if (1e-9 > abs(dp[curId][curK][curS] - max(dp[subId][curK-a][curS-1], load))) {
                    // found the next stage
                    found = true;

                    ResultStage rs;
                    rs.devicesForStage = a;
                    rs.nodes = vectorOfSetBits(getContiguousSet(curId, subId));
                    result.stages.push_back(rs);

                    dbg << "formed a stage with nodes [" << rs.nodes
                        << "] using a=" << a << " many devices, yielding load=" << load << endl;
                    
                    curS -= 1;
                    curK -= a;
                    curId = subId;
                    
                    break;
                }
            }
            if (found) break;
        }
        if (!found) fail("didn't find any reconstruction step to make?");
    }
    if (curS > 0) fail("s didn't fall to 0 by the end of reconstruction?");
    assert(result.stages.size() == finalS);
    // curK, however, might not be 0 by the end
    dbg << "solution used " << ins.maxDevices / finalD - curK
        << " out of available " << ins.maxDevices / finalD << " devices per pipeline" << endl;
    dbg << "data parallel degree: " << finalD << endl;
    dbg << "tensor parallel degree: " << tmpcWidth << endl;
    dbg << "number of stages: " << finalS << endl;
    dbg << endl;

    dbg << "time spent: " << endl;
    dbg << "  in getLoadOfStage: " << timeSpentInGetLoadOfStage << endl;
    dbg << "  in (rest of) DP loop: " << timeSpentInDPLoop << endl;
    dbg << "  in (rest of) final DP round: " << timeSpentInFinalDPRound << endl;

    const double finalTimePerBatchSanityCheck = getTimePerBatchForResult(result);
    if (abs(finalTimePerBatch - finalTimePerBatchSanityCheck) > 1e-9) {
        dbg << "finalTimePerBatch = " << finalTimePerBatch << endl;
        dbg << "finalTimePerBatchSanityCheck = " << finalTimePerBatchSanityCheck << endl;
        fail("finalTimePerBatch != finalTimePerBatchSanityCheck");
    }

    return result;
}


map<int,LoadOfStage> Graph::getLoadOfStage (int id, int subId) const {
    vector<bool> nodesVB = getContiguousSet(id, subId);
    vector<int> nodes = vectorOfSetBits(nodesVB);
    return getLoadOfStage(nodes, nodesVB);
}


map<int,LoadOfStage> Graph::getLoadOfStage (const vector<int> &nodes) const {
    vector<bool> nodesVB = vector<bool>(ins.nodes.at(tmpcWidth).size(), false);
    for (int v : nodes) nodesVB[v] = true;
    return getLoadOfStage(nodes, nodesVB);
}


// Start running the greedy algorithm to place the nodes in the stage
// onto a accelerators. If at any point we find that having more accelerators might
// be useful, we split off another execution with a+1.
// At the end, save the result as resultMap[a].

// For now, just a version that does not handle any layer-level branching,
// but it does handle TMPCs.
// (It assumes it is run for `a` being enough to do tensor-parallel computation,
//  if there is a tensor-parallelized layer in `nodes`.)
void Graph::getLoadOfStageForA (const vector<int> &nodes, const vector<bool> &nodesVB,
                                int a, map<int,LoadOfStage> &resultMap) const {
    // this better be deterministic, as it is run again during reconstruction

    // FW pass
    unordered_map<int,double> finishingTime;
    double fwPassLatency = 0.0;
    int prevNode = -1;
    for (int v : nodes) {
        // nodes in `nodes` come in topological order

        // when can we start running v?
        double startTime = 0.0;
        bool seenEdgeFromPrevNode = false;
        for (const pair<int,double>& it : incomingEdges.at(v)) {
            if (prevNode == it.first) {
                seenEdgeFromPrevNode = true;
            }
            if (finishingTime.count(it.first)) {
                // in our simple special case, the wanted tensor will already be on this device,
                // so just take the finishing time of it.first into accout
                startTime = max(startTime, finishingTime[it.first]);
            } else {
                // the wanted tensor was available already at time 0, but on some other device,
                // so need to take transfer cost into account.
                // a somewhat optimistic "communication delay" computation
                // (does not model contention between transfers over different edges, for example)
                startTime = max(startTime, it.second / ins.bandwidth);
            }
        }
        if (prevNode != -1 && !seenEdgeFromPrevNode) fail("branching in the layer graph, but this is not supported yet");
        finishingTime[v] = startTime + node[v]->optimalLatencyFw;
        fwPassLatency = finishingTime[v];
        prevNode = v;
    }
    // we ignore the outgoing edges - these are the following stages' problem, not ours
    // fwPassLatency computed

    // BW pass (standalone, without recomputing FW pass)
    finishingTime.clear();
    double bwPassLatency = 0.0;
    for (int j = nodes.size()-1; j >= 0; --j) {
        const int v = nodes[j];

        // when can we start running v?
        double startTime = 0.0;
        for (const pair<int,double>& it : outgoingEdges.at(v)) {
            // same as above
            if (finishingTime.count(it.first)) {
                startTime = max(startTime, finishingTime[it.first]);
            } else {
                startTime = max(startTime, it.second / ins.bandwidth);
            }
        }
        finishingTime[v] = startTime + node[v]->optimalLatencyBw;
        bwPassLatency = finishingTime[v];
    }
    // bwPassLatency computed

    // now FW+BW pass (when doing activation recomputation)
    finishingTime.clear();
    double fwBwPassLatency_fw_part = 0.0;
    for (int v : nodes) {
        double startTime = 0.0;
        for (const pair<int,double>& it : incomingEdges.at(v)) {
            // same as above
            // but now, if it.first is not in this stage, then the wanted tensor is already stashed
            // on this accelerator, so we do not need to take this into account
            if (finishingTime.count(it.first)) {
                startTime = max(startTime, finishingTime[it.first]);
            }
        }
        finishingTime[v] = startTime + node[v]->optimalLatencyFw;
        fwBwPassLatency_fw_part = finishingTime[v];
    }
    finishingTime.clear();
    double fwBwPassLatency_entire = 0.0;
    for (int j = nodes.size()-1; j >= 0; --j) {
        const int v = nodes[j];
        double startTime = fwBwPassLatency_fw_part;
        for (const pair<int,double>& it : outgoingEdges.at(v)) {
            if (finishingTime.count(it.first)) {
                startTime = max(startTime, finishingTime[it.first]);
            } else {
                // the wanted (backward) tensor is coming from another device,
                // but it could already begin the transfer at time 0
                // (during our forward pass)
                startTime = max(startTime, it.second / ins.bandwidth);
            }
        }
        finishingTime[v] = startTime + node[v]->optimalLatencyBw;
        fwBwPassLatency_entire = finishingTime[v];
    }

    LoadOfStage result;
    result.fw_bw_latency_wo_recompute = fwPassLatency + bwPassLatency;
    result.fw_bw_latency_with_recompute = fwPassLatency + fwBwPassLatency_entire;
    result.parameter_size = 0.0;
    for (int v : nodes) {
        result.parameter_size += node[v]->parameterSize;
    }
    // left to compute the memory usage,
    // which is of the form const_memory_usage + (s-1) * stashed_data
    const double optimizerSize = (ins.optimizerAlgorithm == "Adam") ? result.parameter_size : 0;
    double activationsSize = 0.0;
    for (int v : nodes) {
        activationsSize += node[v]->activationSize;
        // these are ALL the intermediate activations
    }
    const double constMemoryUsage = 2 * result.parameter_size + optimizerSize + activationsSize;
    double stashedData = 0.0;
    if (ins.activationRecomputation) {
        // stash all activations that come over incoming edges
        for (int v : nodes) {
            for (const pair<int,double>& it : incomingEdges.at(v)) {
                if (!nodesVB[it.first]) {
                    // edge is coming from a previous stage
                    stashedData += it.second;
                }
            }
        }
    } else {
        // stash all activations
        stashedData = activationsSize;
    }
    // okay, now we want to return the largest s such that constMemoryUsage + (s-1) * stashedData <= ins.maxMemoryPerDevice
    // first check if s = ins.maxDevices would be fine
    if (constMemoryUsage + (ins.maxDevices-1) * stashedData <= ins.maxMemoryPerDevice) {
        result.max_s_memory_feasible = ins.maxDevices; // we never need more than that
    } else if (constMemoryUsage > ins.maxMemoryPerDevice) {
        // even s = 1 is impossible
        result.max_s_memory_feasible = 0;
    } else {
        // something in between
        result.max_s_memory_feasible = 1 + floor((ins.maxMemoryPerDevice - constMemoryUsage) / stashedData);
    }

    // NOTE: the formula constMemoryUsage + (s-1) * stashedData is applicable to
    // pipelining schemes such as PipeDream-Flush (1F1B). For GPipe,
    // a more appropriate formula is constMemoryUsage + num_microbatches_per_pipeline * stashedData,
    // but num_microbatches_per_pipeline is mbsInBatch / d,
    // and so we would have to make the DP aware of d, which is not the case right now.
    // (Time complexity wise, it should actually be fine, but requires some changes.)
    // So for now we do not support GPipe

    // all good
    resultMap[a] = result;

    // TODO: possible optimization: adding one more node at the end should be easy
    // (i.e. extending the contiguous subgraph by one node should "fork" from here, not rerun everything)
}


map<int,LoadOfStage> Graph::getLoadOfStage (const vector<int> &nodes, const vector<bool> &nodesVB) const {
    // this better be deterministic, as it is run again during reconstruction

    // by the consistent Megatron assumption, if there is a TMPC-able node
    // in this stage, we need to TMPC it, so then we need a >= t
    int startingA = 1;
    for (int v : nodes) {
        if (node[v]->isTensorParallelized) {
            startingA = tmpcWidth;
            break;
        }
    }
    map<int,LoadOfStage> result;
    getLoadOfStageForA(nodes, nodesVB, startingA, result);
    return result;
}


void Graph::renumberResultBack (Result &r) const {
    for (ResultStage &rs : r.stages) {
        for (int &nodeId : rs.nodes) {
            nodeId = ins.oldNumber[nodeId];
        }
    }
}

double Graph::getTimePerBatchForResult (const Result &r) const {
    // for sanity checks of returned solutions

    if (r.stages.empty()) {
        // infeasible/OOM/empty result
        return INFTY;
    }

    // check that the solution is contiguous
    // (and that there is some topological order in the contracted graph)
    // and that every node belongs to exactly one subgraph
    vector<int> stageOfNode(node.size(), -1);
    int devicesUsedPerPipeline = 0;
    for (int i = 0; i < r.stages.size(); ++i) {
        for (int v : r.stages[i].nodes) {
            if (stageOfNode[v] != -1) {
                fail("duplicate node");
            }
            stageOfNode[v] = i;
        }
        if (r.stages[i].devicesForStage < 1 || r.stages[i].devicesForStage > ins.maxDevices) {
            fail("wrong number of devices for stage");
        }
        devicesUsedPerPipeline += r.stages[i].devicesForStage;
    }
    for (const Edge &e : ins.edges.at(tmpcWidth)) {
        if (stageOfNode[e.sourceId] > stageOfNode[e.destId]) {
            printf("edge %d -> %d\n", e.sourceId, e.destId);
            fail("problem with contiguity (or stages given in wrong order)");
        }
    }
    for (int v = 0; v < ins.nodes.at(tmpcWidth).size(); ++v) {
        if (-1 == stageOfNode[v]) {
            fail("node does not appear in any subgraph");
        }
    }
    if (r.dataParallelDegree < 1 || r.dataParallelDegree > ins.maxDevices || r.dataParallelDegree > ins.mbsInBatch) {
        fail("wrong data-parallel degree");
    }
    if (DATA_PARALLEL_DEGREE_MUST_DIVIDE_NUM_DEVICES &&
        ins.maxDevices % r.dataParallelDegree != 0) {
        fail("data-parallel degree must divide the number of devices");
    }
    if (DATA_PARALLEL_DEGREE_MUST_DIVIDE_BATCH_SIZE &&
        ins.mbsInBatch % r.dataParallelDegree != 0) {
        fail("data-parallel degree must divide the number of microbatches in a batch");
    }
    if (r.tensorParallelDegree != tmpcWidth) {
        fail("wrong tensor-parallel degree (using wrong graph?)");
    }
    if (devicesUsedPerPipeline * r.dataParallelDegree > ins.maxDevices) {
        fail("too many devices used");
    }

    // compute the load, and check if memory usage is okay
    double maxLoad = 0.0;
    double parameterSizeInFirstLayer = -1.0;
    for (int stage = 0; stage < r.stages.size(); ++stage) {
        const int s = r.stages.size() - stage; // which stage this is, counting from the end
        const int a = r.stages[stage].devicesForStage;

        // compute the load of this stage
        map<int,LoadOfStage> loadOfStage = getLoadOfStage(r.stages[stage].nodes);

        // the consistent Megatron assumption is verified inside getLoadOfStage

        // now, we want to get loadOfStage.at(a), but it could be that a is unnecessarily large
        // (i.e. a-1 devices would be enough as well to get the same load, and thus a is not in the map)
        // so we need to find the largest key in the map that is <= a
        int bestA = -1;
        for (const pair<int,LoadOfStage> &it : loadOfStage) {
            if (it.first <= a) {
                bestA = it.first;
            }
        }
        if (bestA == -1) {
            fail("no feasible solution using at most " + to_string(a) + " devices found for this stage");
        }
        if (bestA != a) {
            dbg << "WARNING: would be enough to use " << bestA << " devices instead of "
                << a << " for this stage" << endl;
        }
        if (stage == 0) {
            parameterSizeInFirstLayer = loadOfStage.at(bestA).parameter_size;
        }

        // check memory usage
        if (s > loadOfStage.at(bestA).max_s_memory_feasible) {
            fail("memory usage too high");
        }

        const double load = (ins.activationRecomputation && s > 1)
                            ? loadOfStage.at(bestA).fw_bw_latency_with_recompute
                            : loadOfStage.at(bestA).fw_bw_latency_wo_recompute;
        maxLoad = max(maxLoad, load);
    }

    // maxLoad is computed, now compute timePerBatch
    const int mbsInBatchPerPipeline = ceildiv(ins.mbsInBatch, r.dataParallelDegree);
    const double timePerBatch = maxLoad * (mbsInBatchPerPipeline + r.stages.size() - 1)
                + getDataParallelResyncCost(r.dataParallelDegree, parameterSizeInFirstLayer);
    return timePerBatch;
}


// returns RENUMBERED-back result
pair<Result,double> run (const Instance &ins) {
    double bestTPB = INFTY;
    Result bestResult;
    for (const auto &it : ins.nodes) {
        int tmpcWidth = it.first;
        dbg << "building Graph for tmpcWidth = " << tmpcWidth << endl;
        Graph g(ins, tmpcWidth);
        Result r = g.runDP();
        double tpb = g.getTimePerBatchForResult(r);
        dbg << "TPB = " << tpb << " for tmpcWidth = " << tmpcWidth << endl;
        if (tpb < bestTPB) {
            bestTPB = tpb;
            g.renumberResultBack(r);
            bestResult = r;
        }
    }
    return make_pair(bestResult, bestTPB);
}


double fixedConfigTimePerBatch(const Instance &ins) {
    dbg << endl << "now trying manual partitioning..." << endl;
    Result r;
    r.tensorParallelDegree = ins.fixedDPStrategy[1];
    r.dataParallelDegree = ins.fixedDPStrategy[2];
    int depthOfPipeline = ins.fixedDPStrategy[0];
    int numTransformers = ins.numTransformerLayers;

    int numTransformersPerStage = (numTransformers + depthOfPipeline - 1) / depthOfPipeline;

    for (int stage = 0; stage < depthOfPipeline; ++stage) {
        ResultStage rs;
        rs.devicesForStage = r.tensorParallelDegree;
        if (stage == 0) {
            for (const Node &v : ins.nodes.at(r.tensorParallelDegree)) {
                if (v.id < 0) rs.nodes.push_back(v.id); // junk goes to the first stage
            }
        }
        for (int i = 0; i < numTransformers; ++i) {
            if (i/numTransformersPerStage == stage) rs.nodes.push_back(i);
        }
        if (stage == depthOfPipeline-1) {
            for (const Node &v : ins.nodes.at(r.tensorParallelDegree)) {
                if (v.id >= numTransformers) rs.nodes.push_back(v.id); // junk goes to the last stage
            }
        }
        r.stages.push_back(rs);
        dbg << "stage: " << rs.nodes << endl;
    }
    //ins.maxMemoryPerDevice = ?;
    Graph g(ins, r.tensorParallelDegree);
    double tpb = g.getTimePerBatchForResult(r);
    dbg << "manual TPB = " << tpb << endl;
    return tpb;
}


int main (int argc, char **argv) {
    if (argc != 3) {
        fail("usage: device_placement <input file> <output file>");
    }
    const string inputFilename = argv[1];
    const string outputFilename = argv[2];
    json inputJson;
    ifstream inputFile(inputFilename);
    inputFile >> inputJson;
    Instance ins = inputJson.get<Instance>();
    dbg << "read instance" << endl;
    pair<Result,double> r = run(ins);
    dbg << "got result" << endl;
    ofstream outputFile(outputFilename);
    json outputJson = json(r.first);
    outputJson["finalTimePerBatch"] = r.second;

    double fixedStrategyTimePerBatch = fixedConfigTimePerBatch(ins);
    //double fixedStrategyTimePerBatch = 1;

    outputJson["fixedStrategyTimePerBatch"] = fixedStrategyTimePerBatch;
    outputFile << outputJson.dump(4) << endl;
}
