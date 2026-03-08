/**
 * Interface between ABC and AxSim: read AIG/BLIF/Verilog and produce SoA.
 *
 * Build with ABC: define AXSIM_USE_ABC, add ABC include path (e.g. -I path/to/abc/src) and link libabc.
 * If ABC uses namespace, uncomment "using namespace abc" below.
 */

#include "axsim/abc_interface.hpp"
#include "axsim/circuit_soa.hpp"

#ifdef AXSIM_USE_ABC
// ABC headers (include path must point to ABC src root, e.g. -I path/to/abc/src)
#include "base/abc/abc.h"
#include "base/io/ioAbc.h"
// using namespace abc;
#endif

namespace axsim {

CircuitSoA soa_from_abc_file(const char* filename) {
    CircuitSoA soa;
#ifndef AXSIM_USE_ABC
    (void)filename;
    return soa;
#else
    if (!filename || !filename[0]) return soa;

    Abc_Ntk_t* pNtk = nullptr;
    Abc_Ntk_t* pNtkAig = nullptr;

    // Read netlist (auto-detect format from extension)
    Io_FileType_t ft = Io_ReadFileType(const_cast<char*>(filename));
    if (ft == IO_FILE_NONE || ft == IO_FILE_UNKNOWN) return soa;

    pNtk = Io_ReadNetlist(const_cast<char*>(filename), ft, 0);
    if (!pNtk) return soa;

    // Convert to AIG (strash)
    if (!Abc_NtkIsStrash(pNtk)) {
        pNtkAig = Abc_NtkStrash(pNtk, 0, 0, 0);
        Abc_NtkDelete(pNtk);
        pNtk = pNtkAig;
        if (!pNtk) return soa;
    }

    int num_pis = Abc_NtkPiNum(pNtk);
    int num_pos = Abc_NtkPoNum(pNtk);
    Abc_Obj_t* pConst1 = Abc_AigConst1(pNtk);

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

    // Topological order (PIs then ANDs)
    Vec_Ptr_t* vOrder = Abc_NtkDfsReverse(pNtk);
    std::vector<std::tuple<int, int, bool, bool>> and_nodes;
    and_nodes.reserve(Abc_NtkNodeNum(pNtk));

    for (i = 0; i < Vec_PtrSize(vOrder); i++) {
        Abc_Obj_t* pObj = static_cast<Abc_Obj_t*>(Vec_PtrEntry(vOrder, i));
        if (Abc_ObjIsPi(pObj)) continue;
        if (pConst1 && pObj == pConst1) continue;
        if (!Abc_AigNodeIsAnd(pObj)) continue;

        Abc_Obj_t* p0 = Abc_ObjFanin0(pObj);
        Abc_Obj_t* p1 = Abc_ObjFanin1(pObj);
        int c0 = Abc_ObjFaninC0(pObj);
        int c1 = Abc_ObjFaninC1(pObj);
        Abc_Obj_t* r0 = Abc_ObjRegular(p0);
        Abc_Obj_t* r1 = Abc_ObjRegular(p1);
        int id0 = id_to_our[Abc_ObjId(r0)];
        int id1 = id_to_our[Abc_ObjId(r1)];

        // Fold AND(x, Const1) or AND(Const1, x) to x
        if (id0 == -2 || id1 == -2) {
            id_to_our[Abc_ObjId(pObj)] = (id0 == -2) ? id1 : id0;
            continue;
        }

        and_nodes.push_back({id0, id1, c0 != 0, c1 != 0});
        id_to_our[Abc_ObjId(pObj)] = num_pis + static_cast<int>(and_nodes.size()) - 1;
    }
    Vec_PtrFree(vOrder);

    std::vector<int> out_ids;
    std::vector<uint8_t> out_compl(num_pos);
    out_ids.reserve(num_pos);
    Abc_Obj_t* pPo;
    Abc_NtkForEachPo(pNtk, pPo, i) {
        Abc_Obj_t* child = Abc_ObjChild0(pPo);
        Abc_Obj_t* r = Abc_ObjRegular(child);
        int comp = Abc_ObjIsComplement(child);
        int oid = id_to_our[Abc_ObjId(r)];
        if (oid < 0) { Abc_NtkDelete(pNtk); return CircuitSoA(); }
        out_ids.push_back(oid);
        out_compl[i] = comp ? 1 : 0;
    }

    Abc_NtkDelete(pNtk);

    soa = flatten_from_aig(num_pis, and_nodes, out_ids);
    soa.is_output_complemented = std::move(out_compl);
    return soa;
#endif
}

} // namespace axsim
