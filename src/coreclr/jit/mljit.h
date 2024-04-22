// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

#ifndef _MLJIT_H
#define _MLJIT_H

#ifdef DEBUG

#include "tensorflow\c\c_api.h"
#include "compiler.h"

class MLJIT_Session
{
public:
    MLJIT_Session();

    TF_Graph*          graph;
    TF_Status*         status;
    TF_SessionOptions* sessionOpts;
    TF_Session*        session;

    int         numInputs;
    TF_Input*   input;
    TF_Tensor** inputValues;

    int         numOutputs;
    TF_Output*  output;
    TF_Tensor** outputValues;
};

#define MLJIT_SET_SCALAR_INPUT_API(NAME, TYPE, INDEX) \
    void SetInput_##NAME(##TYPE value) \
    { \
        assert(strcmp(TF_OperationName(input[INDEX].oper), "action_"#NAME) == 0); \
        ##TYPE* data = reinterpret_cast<##TYPE*>(TF_TensorData(inputValues[INDEX])); \
        *data = value; \
    } \

class MLJIT_Session_CSE : public MLJIT_Session
{
public:
    MLJIT_SET_SCALAR_INPUT_API(cse_cost_ex, float, 0);
    MLJIT_SET_SCALAR_INPUT_API(cse_use_count_weighted_log, int64_t, 1);
    MLJIT_SET_SCALAR_INPUT_API(cse_def_count_weighted_log, int64_t, 2);
    MLJIT_SET_SCALAR_INPUT_API(cse_cost_sz, int64_t, 3);
    MLJIT_SET_SCALAR_INPUT_API(cse_use_count, int64_t, 4);
    MLJIT_SET_SCALAR_INPUT_API(cse_def_count, int64_t, 5);
    MLJIT_SET_SCALAR_INPUT_API(cse_is_live_across_call, int64_t, 6);
    MLJIT_SET_SCALAR_INPUT_API(cse_is_int, int64_t, 7);
    MLJIT_SET_SCALAR_INPUT_API(cse_is_constant_not_shared, int64_t, 8);
    MLJIT_SET_SCALAR_INPUT_API(cse_is_shared_constant, int64_t, 9);
    MLJIT_SET_SCALAR_INPUT_API(cse_cost_is_MIN_CSE_COST, int64_t, 10);
    MLJIT_SET_SCALAR_INPUT_API(cse_is_constant_live_across_call, int64_t, 11);
    MLJIT_SET_SCALAR_INPUT_API(cse_is_constant_min_cost, int64_t, 12);
    MLJIT_SET_SCALAR_INPUT_API(cse_cost_is_MIN_CSE_COST_live_across_call, int64_t, 13);
    MLJIT_SET_SCALAR_INPUT_API(cse_is_GTF_MAKE_CSE, int64_t, 14);
    MLJIT_SET_SCALAR_INPUT_API(cse_num_distinct_locals, int64_t, 15);
    MLJIT_SET_SCALAR_INPUT_API(cse_num_local_occurrences, int64_t, 16);
    MLJIT_SET_SCALAR_INPUT_API(cse_has_call, int64_t, 17);
    MLJIT_SET_SCALAR_INPUT_API(log_cse_use_count_weighted_times_cost_ex, int64_t, 18);
    MLJIT_SET_SCALAR_INPUT_API(log_cse_use_count_weighted_times_num_local_occurrences, int64_t, 19);
    MLJIT_SET_SCALAR_INPUT_API(cse_distance, float, 20);
    MLJIT_SET_SCALAR_INPUT_API(cse_is_containable, int64_t, 21);
    MLJIT_SET_SCALAR_INPUT_API(cse_is_cheap_containable, int64_t, 22);
    MLJIT_SET_SCALAR_INPUT_API(cse_is_live_across_call_in_LSRA_ordering, int64_t, 23);
    MLJIT_SET_SCALAR_INPUT_API(log_pressure_estimated_weight, int64_t, 24);
    MLJIT_SET_SCALAR_INPUT_API(reward, float, 25);
    MLJIT_SET_SCALAR_INPUT_API(step_type, int, 26);
    MLJIT_SET_SCALAR_INPUT_API(discount, float, 27);

    int64_t GetOutput_cse_decision()
    {
        assert(strcmp(TF_OperationName(output[0].oper), "StatefulPartitionedCall") == 0);
        int64_t* data = reinterpret_cast<int64_t*>(TF_TensorData(outputValues[0]));
        return data[0];
    }
};

MLJIT_Session_CSE* mljit_session_create_cse();

void mljit_session_destroy(MLJIT_Session* mljitSession);

void mljit_session_action(MLJIT_Session* mljitSession);

#endif // DEBUG

#endif // _MLJIT_H
