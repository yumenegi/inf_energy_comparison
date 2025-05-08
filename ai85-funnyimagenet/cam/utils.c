/******************************************************************************
 *
 * Copyright (C) 2022-2023 Maxim Integrated Products, Inc. (now owned by 
 * Analog Devices, Inc.),
 * Copyright (C) 2023-2024 Analog Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ******************************************************************************/

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "mxc_device.h"
#include "led.h"
#include "board.h"
#include "mxc_delay.h"
#include "uart.h"
#include "rtc.h"
#include "utils.h"

#pragma GCC optimize("-O0")

#define DEBUG_COMPORT MXC_UART0

/***************************** VARIABLES *************************************/

/************************    PUBLIC FUNCTIONS  *******************************/
void utils_delay_ms(uint32_t ms)
{
    MXC_Delay(ms * 1000UL);
}

uint32_t utils_get_time_ms(void)
{
    uint32_t sec, ssec;
    double subsec;
    uint32_t ms;

    MXC_RTC_GetSubSeconds(&ssec);
    subsec = (double)ssec / (double)(4096.0);

    MXC_RTC_GetSeconds(&sec);

    ms = (sec * 1000) + (int)(subsec * 1000);

    return ms;
}

void utils_hexDump(const char *title, uint8_t *buf, uint32_t len)
{
    uint32_t i;

    // Display the title
    if (title) {
        printf("%s", title);
    }

    // Display the buffer bytes
    for (i = 0; i < len; i++) {
        if (!(i % 16)) {
            printf("\n");
        }

        printf("%02X ", buf[i]);
    }

    printf("\n");
}

static void utils_send_byte(mxc_uart_regs_t *uart, uint8_t value)
{
    while (MXC_UART_WriteCharacter(uart, value) == E_OVERFLOW) {}
}

static void utils_send_bytes(mxc_uart_regs_t *uart, uint8_t *ptr, int length)
{
    int i;

    for (i = 0; i < length; i++) {
        utils_send_byte(uart, ptr[i]);
    }
}

int utils_send_img_to_pc(uint8_t *img, uint32_t imgLen, int w, int h, uint8_t *pixelformat)
{
    int len;

    // Transmit the start token
    len = 5;
    utils_send_bytes(DEBUG_COMPORT, (uint8_t *)"*STR*", len);

    // Transmit the width of the image
    utils_send_byte(DEBUG_COMPORT, (w >> 8) & 0xff); // high byte
    utils_send_byte(DEBUG_COMPORT, (w >> 0) & 0xff); // low byte
    // Transmit the height of the image
    utils_send_byte(DEBUG_COMPORT, (h >> 8) & 0xff); // high byte
    utils_send_byte(DEBUG_COMPORT, (h >> 0) & 0xff); // low byte

    // Transmit the pixel format of the image
    len = strlen((char *)pixelformat);
    utils_send_byte(DEBUG_COMPORT, len & 0xff);
    utils_send_bytes(DEBUG_COMPORT, pixelformat, len);

    // Transmit the image length in bytes
    utils_send_byte(DEBUG_COMPORT, (imgLen >> 24) & 0xff); // high byte
    utils_send_byte(DEBUG_COMPORT, (imgLen >> 16) & 0xff); // low byte
    utils_send_byte(DEBUG_COMPORT, (imgLen >> 8) & 0xff); // low byte
    utils_send_byte(DEBUG_COMPORT, (imgLen >> 0) & 0xff); // low byte

    // Send the image pixel bytes
    while (imgLen) {
        len = imgLen;
        utils_send_bytes(DEBUG_COMPORT, img, len);
        img += len;
        imgLen -= len;
    }

    return 0;
}

int utils_stream_img_to_pc_init(uint8_t *img, uint32_t imgLen, int w, int h, uint8_t *pixelformat)
{
    int len;

    // Transmit the start token
    len = 5;
    utils_send_bytes(DEBUG_COMPORT, (uint8_t *)"*STR*", len);

    // Transmit the width of the image
    utils_send_byte(DEBUG_COMPORT, (w >> 8) & 0xff); // high byte
    utils_send_byte(DEBUG_COMPORT, (w >> 0) & 0xff); // low byte
    // Transmit the height of the image
    utils_send_byte(DEBUG_COMPORT, (h >> 8) & 0xff); // high byte
    utils_send_byte(DEBUG_COMPORT, (h >> 0) & 0xff); // low byte

    // Transmit the pixel format of the image
    len = strlen((char *)pixelformat);
    utils_send_byte(DEBUG_COMPORT, len & 0xff);
    utils_send_bytes(DEBUG_COMPORT, pixelformat, len);

    // Transmit the image length in bytes
    utils_send_byte(DEBUG_COMPORT, (imgLen >> 24) & 0xff); // high byte
    utils_send_byte(DEBUG_COMPORT, (imgLen >> 16) & 0xff); // low byte
    utils_send_byte(DEBUG_COMPORT, (imgLen >> 8) & 0xff); // low byte
    utils_send_byte(DEBUG_COMPORT, (imgLen >> 0) & 0xff); // low byte

    return 0;
}

int utils_stream_image_row_to_pc(uint8_t *img, uint32_t imgRowLen)
{
    // Send the image pixel bytes
    utils_send_bytes(DEBUG_COMPORT, img, imgRowLen);

    return 0;
}
