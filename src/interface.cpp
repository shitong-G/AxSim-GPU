/**
 * Interface between ABC and AxSim: read AIG/BLIF/Verilog and produce SoA.
 *
 * Build with ABC: define AXSIM_USE_ABC, add ABC include path (e.g. -I path/to/abc/src) and link libabc.
 * If ABC uses namespace, uncomment "using namespace abc" below.
 */

#include "axsim/abc_interface.hpp"
#include "axsim/circuit_soa.hpp"
#define AXSIM_USE_ABC
#include <cstdio>

#ifdef AXSIM_USE_ABC
// ABC headers: compiler must have -I path/to/abc/src (paths are relative to that root).
#include "base/abc/abc.h"
#include "base/io/ioAbc.h"
#include "base/main/main.h"
// using namespace abc;
#endif

namespace axsim {

namespace {
CircuitSoA make_invalid_soa() {
    CircuitSoA s;
    s.num_pis = -1;
    return s;
}

#ifdef AXSIM_USE_ABC
void ensure_abc_started() {
    // Io_ReadNetlist on some ABC builds expects the global frame to be initialized.
    static bool started = false;
    if (!started) {
        Abc_Start();
        started = true;
    }
}
#endif
} // namespace

CircuitSoA soa_from_abc_file(const char* filename) {
    CircuitSoA soa = make_invalid_soa();
#ifndef AXSIM_USE_ABC
    (void)filename;
    fprintf(stderr, "[soa_from_abc_file] ABC support is disabled at compile time.\n");
    return soa;
#else
    ensure_abc_started();

    if (!filename || !filename[0]) {
        fprintf(stderr, "[soa_from_abc_file] empty filename.\n");
        return soa;
    }
    fprintf(stderr, "[soa_from_abc_file] loading '%s'\n", filename);

    Abc_Ntk_t* pNtk = nullptr;
    Abc_Ntk_t* pNtkAig = nullptr;

    // Read netlist (auto-detect format from extension)
    Io_FileType_t ft = Io_ReadFileType(const_cast<char*>(filename));
    if (ft == IO_FILE_NONE || ft == IO_FILE_UNKNOWN) {
        fprintf(stderr, "[soa_from_abc_file] unsupported/unknown file type for '%s'\n", filename);
        return soa;
    }

    pNtk = Io_ReadNetlist(const_cast<char*>(filename), ft, 0);
    if (!pNtk) {
        fprintf(stderr, "[soa_from_abc_file] Io_ReadNetlist failed for '%s'\n", filename);
        return soa;
    }

    // Convert to logic if needed, then to AIG (strash).
    if (!Abc_NtkIsStrash(pNtk)) {
        if (!Abc_NtkIsLogic(pNtk)) {
            Abc_Ntk_t* pNtkLogic = Abc_NtkToLogic(pNtk);
            Abc_NtkDelete(pNtk);
            pNtk = pNtkLogic;
            if (!pNtk) {
                fprintf(stderr, "[soa_from_abc_file] Abc_NtkToLogic failed for '%s'\n", filename);
                return soa;
            }
        }
        pNtkAig = Abc_NtkStrash(pNtk, 0, 0, 0);
        Abc_NtkDelete(pNtk);
        pNtk = pNtkAig;
        if (!pNtk) {
            fprintf(stderr, "[soa_from_abc_file] Abc_NtkStrash failed for '%s'\n", filename);
            return soa;
        }
    }

    int num_pis = Abc_NtkPiNum(pNtk);
    int num_pos = Abc_NtkPoNum(pNtk);
    Abc_Obj_t* pConst1 = Abc_AigConst1(pNtk);
    fprintf(stderr, "[soa_from_abc_file] stats: PIs=%d POs=%d AIG nodes=%d maxObjId=%d\n",
            num_pis, num_pos, Abc_NtkNodeNum(pNtk), Abc_NtkObjNumMax(pNtk));

    // Map: ABC Obj Id -> our node index. -1 unset, -2 Const1 (will fold AND(x,1)→x)
    int maxId = Abc_NtkObjNumMax(pNtk);
    std::vector<int> id_to_our(maxId, -1);

    int our_id = 0;
    Abc_Obj_t* pPi;
    int i;
    Abc_NtkForEachPi(pNtk, pPi, i)
        id_to_our[Abc_ObjId(pPi)] = our_id++;
    if (pConst1)
        id_to_our[Abc_ObjId(pConst1)] = -2;

    // Collect candidate AND nodes first; then resolve them in dependency order.
    Vec_Ptr_t* vOrder = Abc_NtkDfsReverse(pNtk);
    std::vector<std::tuple<int, int, bool, bool>> and_nodes;
    and_nodes.reserve(Abc_NtkNodeNum(pNtk));
    std::vector<Abc_Obj_t*> pending_ands;
    pending_ands.reserve(Abc_NtkNodeNum(pNtk));

    for (i = 0; i < Vec_PtrSize(vOrder); i++) {
        Abc_Obj_t* pObj = static_cast<Abc_Obj_t*>(Vec_PtrEntry(vOrder, i));
        if (Abc_ObjIsPi(pObj)) continue;
        if (pConst1 && pObj == pConst1) continue;
        if (!Abc_AigNodeIsAnd(pObj)) continue;
        pending_ands.push_back(pObj);
    }
    Vec_PtrFree(vOrder);

    int pass = 0;
    while (!pending_ands.empty()) {
        ++pass;
        size_t resolved_this_pass = 0;
        std::vector<Abc_Obj_t*> next_pending;
        next_pending.reserve(pending_ands.size());

        for (Abc_Obj_t* pObj : pending_ands) {
            Abc_Obj_t* p0 = Abc_ObjFanin0(pObj);
            Abc_Obj_t* p1 = Abc_ObjFanin1(pObj);
            int c0 = Abc_ObjFaninC0(pObj);
            int c1 = Abc_ObjFaninC1(pObj);
            Abc_Obj_t* r0 = Abc_ObjRegular(p0);
            Abc_Obj_t* r1 = Abc_ObjRegular(p1);
            int id0 = id_to_our[Abc_ObjId(r0)];
            int id1 = id_to_our[Abc_ObjId(r1)];

            // Need both fanins resolved (or Const1 marker) before materializing this AND.
            if ((id0 < 0 && id0 != -2) || (id1 < 0 && id1 != -2)) {
                next_pending.push_back(pObj);
                continue;
            }

            // Fold AND(x, Const1) or AND(Const1, x) to x
            if (id0 == -2 || id1 == -2) {
                id_to_our[Abc_ObjId(pObj)] = (id0 == -2) ? id1 : id0;
            } else {
                and_nodes.push_back({id0, id1, c0 != 0, c1 != 0});
                id_to_our[Abc_ObjId(pObj)] = num_pis + static_cast<int>(and_nodes.size()) - 1;
            }
            ++resolved_this_pass;
        }

        // fprintf(stderr, "[soa_from_abc_file] resolve pass %d: resolved=%zu pending=%zu\n",
        //         pass, resolved_this_pass, next_pending.size());

        if (resolved_this_pass == 0) {
            fprintf(stderr, "[soa_from_abc_file] stalled while resolving AND dependencies in '%s'\n", filename);
            for (size_t k = 0; k < next_pending.size() && k < 8; ++k) {
                Abc_Obj_t* pObj = next_pending[k];
                Abc_Obj_t* r0 = Abc_ObjRegular(Abc_ObjFanin0(pObj));
                Abc_Obj_t* r1 = Abc_ObjRegular(Abc_ObjFanin1(pObj));
                int id0 = id_to_our[Abc_ObjId(r0)];
                int id1 = id_to_our[Abc_ObjId(r1)];
                fprintf(stderr, "  unresolved AND obj=%d fanin_obj=(%d,%d) mapped=(%d,%d)\n",
                        Abc_ObjId(pObj), Abc_ObjId(r0), Abc_ObjId(r1), id0, id1);
            }
            Abc_NtkDelete(pNtk);
            return make_invalid_soa();
        }

        pending_ands.swap(next_pending);
    }

    std::vector<int> out_ids;
    std::vector<uint8_t> out_compl(num_pos);
    out_ids.reserve(num_pos);
    int const0_id = -1; // lazily materialized as (pi0 & ~pi0)
    Abc_Obj_t* pPo;
    Abc_NtkForEachPo(pNtk, pPo, i) {
        Abc_Obj_t* child = Abc_ObjChild0(pPo);
        Abc_Obj_t* r = Abc_ObjRegular(child);
        int comp = Abc_ObjIsComplement(child);
        int oid = id_to_our[Abc_ObjId(r)];
        if (oid == -2) {
            // ABC uses const1 (+ optional edge complement for const0).
            // Materialize one const0 node as (pi0 & ~pi0), then const1 = ~const0.
            if (const0_id < 0) {
                if (num_pis <= 0) {
                    fprintf(stderr, "[soa_from_abc_file] cannot materialize const output without PI in '%s' (po=%d)\n",
                            filename, i);
                    Abc_NtkDelete(pNtk);
                    return make_invalid_soa();
                }
                and_nodes.push_back({0, 0, false, true}); // pi0 & ~pi0 = 0
                const0_id = num_pis + static_cast<int>(and_nodes.size()) - 1;
                fprintf(stderr, "[soa_from_abc_file] materialized const0 as node id=%d for '%s'\n",
                        const0_id, filename);
            }
            out_ids.push_back(const0_id);
            out_compl[i] = comp ? 0 : 1; // child==const1 => ~const0; child==~const1(const0) => const0
            continue;
        }
        if (oid < 0) {
            fprintf(stderr, "[soa_from_abc_file] PO mapping failed in '%s': po_idx=%d po_obj=%d child_obj=%d regular_obj=%d mapped_oid=%d comp=%d\n",
                    filename, i, Abc_ObjId(pPo), Abc_ObjId(child), Abc_ObjId(r), oid, comp);
            Abc_NtkDelete(pNtk);
            return make_invalid_soa();
        }
        out_ids.push_back(oid);
        out_compl[i] = comp ? 1 : 0;
    }

    Abc_NtkDelete(pNtk);

    soa = flatten_from_aig(num_pis, and_nodes, out_ids);
    soa.is_output_complemented = std::move(out_compl);
    fprintf(stderr, "[soa_from_abc_file] built SoA: num_pis=%d num_nodes=%d num_ands=%d num_outputs=%d valid=%d\n",
            soa.num_pis, soa.num_nodes, soa.num_ands, soa.num_outputs, soa.valid() ? 1 : 0);
    return soa;
#endif
}

} // namespace axsim
