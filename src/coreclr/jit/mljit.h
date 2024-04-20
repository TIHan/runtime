// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

#ifndef _MLJIT_H
#define _MLJIT_H

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

MLJIT_Session* mljit_session_create_cse();

void mljit_session_destroy(MLJIT_Session* mljitSession);

void mljit_session_action(MLJIT_Session* mljitSession);

#endif // _MLJIT_H
