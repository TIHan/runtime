// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

#ifndef _MLJIT_H
#define _MLJIT_H

#define MLJIT 1

#ifdef MLJIT
#define MLJIT_ARG(x)  , x
#define MLJIT_ARG1(x) x
#define MLJITARG(x)   , x
#else
#define MLJIT_ARG(x) DEBUG_ARG(x)
#define MLJIT_ARG1(x) DEBUG_ARG1(x)
#define MLJITARG      DEBUGARG(x)
#endif

#ifdef MLJIT

#include "tensorflow\c\c_api.h"
#include "compiler.h"

struct MLJIT_float_2
{
    float values[2];
};

struct MLJIT_CseActionLogItem
{
    int64_t cse_index;
    float cse_cost_ex;
    float cse_use_count_weighted_log;
    float cse_def_count_weighted_log;
    int64_t cse_cost_sz;
    int64_t cse_use_count;
    int64_t cse_def_count;
    int64_t cse_is_live_across_call;
    int64_t cse_is_int;
    int64_t cse_is_constant_not_shared;
    int64_t cse_is_shared_constant;
    int64_t cse_cost_is_MIN_CSE_COST;
    int64_t cse_is_constant_live_across_call;
    int64_t cse_is_constant_min_cost;
    int64_t cse_cost_is_MIN_CSE_COST_live_across_call;
    int64_t cse_is_GTF_MAKE_CSE;
    int64_t cse_num_distinct_locals;
    int64_t cse_num_local_occurrences;
    int64_t cse_has_call;
    float log_cse_use_count_weighted_times_cost_ex;
    float log_cse_use_count_weighted_times_num_local_occurrences;
    float cse_distance;
    int64_t cse_is_containable;
    int64_t cse_is_cheap_containable;
    int64_t cse_is_live_across_call_in_LSRA_ordering;
    int64_t log_pressure_estimated_weight;
    int64_t cse_decision;
    float   reward;
    MLJIT_float_2 CategoricalProjectionNetwork_logits;
};

class MLJIT_Policy
{
public:
    MLJIT_Policy();

    void Action();

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
    ##TYPE GetInput_##NAME() \
    { \
        assert(strcmp(TF_OperationName(input[INDEX].oper), "action_"#NAME) == 0); \
        ##TYPE* data = reinterpret_cast<##TYPE*>(TF_TensorData(inputValues[INDEX])); \
        return *data; \
    } \

#define MLJIT_RECORD_INPUT(NAME) \
    l.##NAME = GetInput_##NAME(); \

#define MLJIT_RECORD_OUTPUT(NAME) \
    l.##NAME = GetOutput_##NAME(); \

#define MLJIT_WRITE_JSON_PROPERTY_FLOAT(FPTR, NAME) \
    fprintf(FPTR, "\"" #NAME "\": %f,", l->##NAME); \

#define MLJIT_WRITE_JSON_PROPERTY_INT64(FPTR, NAME) \
    fprintf(FPTR, "\"" #NAME "\": %zd,", l->##NAME); \

#define MLJIT_WRITE_JSON_PROPERTY_FLOAT_NO_COMMA(FPTR, NAME) \
    fprintf(FPTR, "\"" #NAME "\": %f", l->##NAME); \

class MLJIT_CsePolicyBase : public MLJIT_Policy
{
public:
    int                    loggedActionCount = 0;
    MLJIT_CseActionLogItem loggedActions[256];

    MLJIT_SET_SCALAR_INPUT_API(cse_index, int64_t, 0);
    MLJIT_SET_SCALAR_INPUT_API(cse_cost_ex, float, 1);
    MLJIT_SET_SCALAR_INPUT_API(cse_use_count_weighted_log, float, 2);
    MLJIT_SET_SCALAR_INPUT_API(cse_def_count_weighted_log, float, 3);
    MLJIT_SET_SCALAR_INPUT_API(cse_cost_sz, int64_t, 4);
    MLJIT_SET_SCALAR_INPUT_API(cse_use_count, int64_t, 5);
    MLJIT_SET_SCALAR_INPUT_API(cse_def_count, int64_t, 6);
    MLJIT_SET_SCALAR_INPUT_API(cse_is_live_across_call, int64_t, 7);
    MLJIT_SET_SCALAR_INPUT_API(cse_is_int, int64_t, 8);
    MLJIT_SET_SCALAR_INPUT_API(cse_is_constant_not_shared, int64_t, 9);
    MLJIT_SET_SCALAR_INPUT_API(cse_is_shared_constant, int64_t, 10);
    MLJIT_SET_SCALAR_INPUT_API(cse_cost_is_MIN_CSE_COST, int64_t, 11);
    MLJIT_SET_SCALAR_INPUT_API(cse_is_constant_live_across_call, int64_t, 12);
    MLJIT_SET_SCALAR_INPUT_API(cse_is_constant_min_cost, int64_t, 13);
    MLJIT_SET_SCALAR_INPUT_API(cse_cost_is_MIN_CSE_COST_live_across_call, int64_t, 14);
    MLJIT_SET_SCALAR_INPUT_API(cse_is_GTF_MAKE_CSE, int64_t, 15);
    MLJIT_SET_SCALAR_INPUT_API(cse_num_distinct_locals, int64_t, 16);
    MLJIT_SET_SCALAR_INPUT_API(cse_num_local_occurrences, int64_t, 17);
    MLJIT_SET_SCALAR_INPUT_API(cse_has_call, int64_t, 18);
    MLJIT_SET_SCALAR_INPUT_API(log_cse_use_count_weighted_times_cost_ex, float, 19);
    MLJIT_SET_SCALAR_INPUT_API(log_cse_use_count_weighted_times_num_local_occurrences, float, 20);
    MLJIT_SET_SCALAR_INPUT_API(cse_distance, float, 21);
    MLJIT_SET_SCALAR_INPUT_API(cse_is_containable, int64_t, 22);
    MLJIT_SET_SCALAR_INPUT_API(cse_is_cheap_containable, int64_t, 23);
    MLJIT_SET_SCALAR_INPUT_API(cse_is_live_across_call_in_LSRA_ordering, int64_t, 24);
    MLJIT_SET_SCALAR_INPUT_API(log_pressure_estimated_weight, int64_t, 25);
    MLJIT_SET_SCALAR_INPUT_API(reward, float, 26);
    MLJIT_SET_SCALAR_INPUT_API(step_type, int, 27);
    MLJIT_SET_SCALAR_INPUT_API(discount, float, 28);

    virtual int64_t GetOutput_cse_decision() = 0;
    virtual void    SetOutput_cse_decision(int64_t value) = 0;

    virtual void LogActionExtra(MLJIT_CseActionLogItem* l) = 0;
    virtual void SaveLoggedActionsAsJsonExtra(FILE* fptr, MLJIT_CseActionLogItem* l) = 0;

    void ResetLog()
    {
        loggedActionCount = 0;
    }

    void LogAction()
    {
        MLJIT_CseActionLogItem l = {};
        MLJIT_RECORD_INPUT(cse_index);
        MLJIT_RECORD_INPUT(cse_cost_ex);
        MLJIT_RECORD_INPUT(cse_use_count_weighted_log);
        MLJIT_RECORD_INPUT(cse_def_count_weighted_log);
        MLJIT_RECORD_INPUT(cse_cost_sz);
        MLJIT_RECORD_INPUT(cse_use_count);
        MLJIT_RECORD_INPUT(cse_def_count);
        MLJIT_RECORD_INPUT(cse_is_live_across_call);
        MLJIT_RECORD_INPUT(cse_is_int);
        MLJIT_RECORD_INPUT(cse_is_constant_not_shared);
        MLJIT_RECORD_INPUT(cse_is_shared_constant);
        MLJIT_RECORD_INPUT(cse_cost_is_MIN_CSE_COST);
        MLJIT_RECORD_INPUT(cse_is_constant_live_across_call);
        MLJIT_RECORD_INPUT(cse_is_constant_min_cost);
        MLJIT_RECORD_INPUT(cse_cost_is_MIN_CSE_COST_live_across_call);
        MLJIT_RECORD_INPUT(cse_is_GTF_MAKE_CSE);
        MLJIT_RECORD_INPUT(cse_num_distinct_locals);
        MLJIT_RECORD_INPUT(cse_num_local_occurrences);
        MLJIT_RECORD_INPUT(cse_has_call);
        MLJIT_RECORD_INPUT(log_cse_use_count_weighted_times_cost_ex);
        MLJIT_RECORD_INPUT(log_cse_use_count_weighted_times_num_local_occurrences);
        MLJIT_RECORD_INPUT(cse_distance);
        MLJIT_RECORD_INPUT(cse_is_containable);
        MLJIT_RECORD_INPUT(cse_is_cheap_containable);
        MLJIT_RECORD_INPUT(cse_is_live_across_call_in_LSRA_ordering);
        MLJIT_RECORD_INPUT(log_pressure_estimated_weight);
        MLJIT_RECORD_OUTPUT(cse_decision);
        MLJIT_RECORD_INPUT(reward);
        LogActionExtra(&l);

        loggedActions[loggedActionCount] = l;
        loggedActionCount++;
    }

    void SaveLoggedActionsAsJson(const WCHAR* path)
    {
        assert(path);

        FILE* fptr;
        fptr = _wfopen(path, W("w"));
        fprintf(fptr, "[");

        for (int i = 0; i < loggedActionCount; i++)
        {
            fprintf(fptr, "{");
            auto l = &loggedActions[i];
            MLJIT_WRITE_JSON_PROPERTY_INT64(fptr, cse_index);
            MLJIT_WRITE_JSON_PROPERTY_FLOAT(fptr, cse_cost_ex);
            MLJIT_WRITE_JSON_PROPERTY_FLOAT(fptr, cse_use_count_weighted_log);
            MLJIT_WRITE_JSON_PROPERTY_FLOAT(fptr, cse_def_count_weighted_log);
            MLJIT_WRITE_JSON_PROPERTY_INT64(fptr, cse_cost_sz);
            MLJIT_WRITE_JSON_PROPERTY_INT64(fptr, cse_use_count);
            MLJIT_WRITE_JSON_PROPERTY_INT64(fptr, cse_def_count);
            MLJIT_WRITE_JSON_PROPERTY_INT64(fptr, cse_is_live_across_call);
            MLJIT_WRITE_JSON_PROPERTY_INT64(fptr, cse_is_int);
            MLJIT_WRITE_JSON_PROPERTY_INT64(fptr, cse_is_constant_not_shared);
            MLJIT_WRITE_JSON_PROPERTY_INT64(fptr, cse_is_shared_constant);
            MLJIT_WRITE_JSON_PROPERTY_INT64(fptr, cse_cost_is_MIN_CSE_COST);
            MLJIT_WRITE_JSON_PROPERTY_INT64(fptr, cse_is_constant_live_across_call);
            MLJIT_WRITE_JSON_PROPERTY_INT64(fptr, cse_is_constant_min_cost);
            MLJIT_WRITE_JSON_PROPERTY_INT64(fptr, cse_cost_is_MIN_CSE_COST_live_across_call);
            MLJIT_WRITE_JSON_PROPERTY_INT64(fptr, cse_is_GTF_MAKE_CSE);
            MLJIT_WRITE_JSON_PROPERTY_INT64(fptr, cse_num_distinct_locals);
            MLJIT_WRITE_JSON_PROPERTY_INT64(fptr, cse_num_local_occurrences);
            MLJIT_WRITE_JSON_PROPERTY_INT64(fptr, cse_has_call);
            MLJIT_WRITE_JSON_PROPERTY_FLOAT(fptr, log_cse_use_count_weighted_times_cost_ex);
            MLJIT_WRITE_JSON_PROPERTY_FLOAT(fptr, log_cse_use_count_weighted_times_num_local_occurrences);
            MLJIT_WRITE_JSON_PROPERTY_FLOAT(fptr, cse_distance);
            MLJIT_WRITE_JSON_PROPERTY_INT64(fptr, cse_is_containable);
            MLJIT_WRITE_JSON_PROPERTY_INT64(fptr, cse_is_cheap_containable);
            MLJIT_WRITE_JSON_PROPERTY_INT64(fptr, cse_is_live_across_call_in_LSRA_ordering);
            MLJIT_WRITE_JSON_PROPERTY_INT64(fptr, log_pressure_estimated_weight);
            MLJIT_WRITE_JSON_PROPERTY_INT64(fptr, cse_decision);
            SaveLoggedActionsAsJsonExtra(fptr, l);
            MLJIT_WRITE_JSON_PROPERTY_FLOAT_NO_COMMA(fptr, reward);
            fprintf(fptr, "}");
            if (i != (loggedActionCount - 1))
            {
                fprintf(fptr, ",");
            }
        }

        fprintf(fptr, "]\n");
        fclose(fptr);
    }
};

class MLJIT_CsePolicy : public MLJIT_CsePolicyBase
{
public:
    int64_t GetOutput_cse_decision() override
    {
        int index = 0; // "0" - policy, "1" - collect_policy
        assert(strcmp(TF_OperationName(output[index].oper), "StatefulPartitionedCall") == 0);
        int64_t* data = reinterpret_cast<int64_t*>(TF_TensorData(outputValues[index]));
        return data[0];
    }

    void SetOutput_cse_decision(int64_t value) override
    {
        int index = 0; // "0" - policy, "1" - collect_policy
        assert(strcmp(TF_OperationName(output[index].oper), "StatefulPartitionedCall") == 0);
        int64_t* data = reinterpret_cast<int64_t*>(TF_TensorData(outputValues[index]));
        data[0]       = value;
    }

    void LogActionExtra(MLJIT_CseActionLogItem* l) override
    {
    }

    void SaveLoggedActionsAsJsonExtra(FILE* fptr, MLJIT_CseActionLogItem* l) override
    {
    }
};

class MLJIT_CseCollectPolicy : public MLJIT_CsePolicyBase
{
public:
    int64_t GetOutput_cse_decision() override
    {
        int index = 1; // "0" - policy, "1" - collect_policy
        assert(strcmp(TF_OperationName(output[index].oper), "StatefulPartitionedCall") == 0);
        int64_t* data = reinterpret_cast<int64_t*>(TF_TensorData(outputValues[index]));
        return data[0];
    }

    void SetOutput_cse_decision(int64_t value) override
    {
        int index = 1; // "0" - policy, "1" - collect_policy
        assert(strcmp(TF_OperationName(output[index].oper), "StatefulPartitionedCall") == 0);
        int64_t* data = reinterpret_cast<int64_t*>(TF_TensorData(outputValues[index]));
        data[0]       = value;
    }

    MLJIT_float_2 GetOutput_CategoricalProjectionNetwork_logits()
    {
        int index = 0; // collect_policy only
        assert(strcmp(TF_OperationName(output[index].oper), "StatefulPartitionedCall") == 0);
        MLJIT_float_2* data = reinterpret_cast<MLJIT_float_2*>(TF_TensorData(outputValues[index]));
        return data[0];
    }

    void LogActionExtra(MLJIT_CseActionLogItem* l) override
    {
        l->CategoricalProjectionNetwork_logits = GetOutput_CategoricalProjectionNetwork_logits();
    }

    void SaveLoggedActionsAsJsonExtra(FILE* fptr, MLJIT_CseActionLogItem* l) override
    {
        auto value1 = l->CategoricalProjectionNetwork_logits.values[0];
        auto value2 = l->CategoricalProjectionNetwork_logits.values[1];
        fprintf(fptr, "\"CategoricalProjectionNetwork_logits\": [%f, %f],", value1, value2);
    }
};

MLJIT_CsePolicyBase* mljit_try_create_cse_policy();
MLJIT_CsePolicyBase* mljit_try_create_cse_collect_policy();
void mljit_destroy_policy(MLJIT_Policy* mljitSession);

#endif // MLJIT

#endif // _MLJIT_H
