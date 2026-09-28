import React, { useEffect, useRef, useState } from 'react'
import {
  AppBar,
  Box,
  IconButton,
  Snackbar,
  Toolbar,
  Typography,
} from '@mui/material'
import {
  DarkMode as DarkModeIcon,
  LightMode as LightModeIcon,
} from '@mui/icons-material'

import ChatInput from './components/ChatInput'
import ChatWindow from './components/ChatWindow'

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'

export default function App({ mode, toggleMode }) {
  const [messages, setMessages] = useState([])
  const [loading, setLoading] = useState(false)
  const [errorMsg, setErrorMsg] = useState('')
  const [streamingMessage, setStreamingMessage] = useState(null)
  const bottomRef = useRef()
  const abortCtrlRef = useRef(null)

  const scrollToBottom = () => bottomRef.current?.scrollIntoView({ behavior: 'smooth' })

  useEffect(() => {
    scrollToBottom()
  }, [messages, loading, streamingMessage])

  const showError = (msg) => setErrorMsg(msg)

  const handleStop = () => {
    if (abortCtrlRef.current) {
      abortCtrlRef.current.abort()
      setLoading(false)
      abortCtrlRef.current = null
    }
    setStreamingMessage(null)
  }

  const streamAssistantReply = (fullText, onDone) => {
    let i = 0
    setStreamingMessage('')
    function nextChar() {
      setStreamingMessage(fullText.slice(0, i + 1))
      i++
      if (i < fullText.length) {
        setTimeout(nextChar, 18)
      } else {
        setMessages((m) => [
          ...m.slice(0, m.length - 1),
          { role: 'assistant', content: fullText }
        ])
        setStreamingMessage(null)
        if (typeof onDone === 'function') onDone()
      }
    }
    nextChar()
  }

  const addAssistantResponse = (data) => {
    if (
      data.type === 'chat' ||
      data.type === 'recommendation' ||
      data.type === 'image-search' ||
      data.type === 'multimodal-search'
    ) {
      setMessages((m) => [
        ...m,
        { role: 'assistant', content: '', isStreaming: true }
      ])
      streamAssistantReply(data.response, () => {
        if (data.products && data.products.length > 0) {
          setMessages((m) => [
            ...m,
            ...data.products.map((p) => ({ product: p }))
          ])
        }
      })
      return true
    }
    return false
  }

  const fileToDataUrl = async (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = (e) => resolve(e.target.result)
      reader.onerror = reject
      reader.readAsDataURL(file)
    })
  }

  const buildProductImageUrl = (imagePath) => {
    if (!imagePath) return null
    if (imagePath.startsWith('http')) return imagePath
    const filename = imagePath.startsWith('images/')
      ? imagePath.slice('images/'.length)
      : imagePath.replace(/^\/+/, '')
    return `${BACKEND_URL}/images/${filename}`
  }

  const sendMessage = async (text, file = null) => {
    if (!text && !file) return

    let previewUrl = null
    if (file) {
      previewUrl = await fileToDataUrl(file)
    }

    setMessages((m) => [
      ...m,
      {
        role: 'user',
        content: file ? { text, image: previewUrl } : text
      }
    ])

    setLoading(true)
    const ctrl = new AbortController()
    abortCtrlRef.current = ctrl

    try {
      let res
      if (file) {
        const form = new FormData()
        form.append('message', text)
        form.append('file', file)
        res = await fetch(`${BACKEND_URL}/chat-multimodal`, {
          method: 'POST',
          body: form,
          signal: ctrl.signal,
        })
      } else {
        res = await fetch(`${BACKEND_URL}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text }),
          signal: ctrl.signal,
        })
      }

      const data = await res.json()
      if (addAssistantResponse(data)) return

      setMessages((m) => [
        ...m,
        { role: 'assistant', content: 'Unexpected backend response.' }
      ])
    } catch (e) {
      if (e.name !== 'AbortError') showError('Failed to contact backend.')
    } finally {
      setLoading(false)
      abortCtrlRef.current = null
    }
  }

  const handleSearchSimilar = async (product) => {
    setMessages((m) => [
      ...m,
      {
        role: 'user',
        content: {
          text: `Find products visually similar to ${product.name}`,
          image: buildProductImageUrl(product.image_path),
        },
      }
    ])

    setLoading(true)
    const ctrl = new AbortController()
    abortCtrlRef.current = ctrl

    try {
      const res = await fetch(`${BACKEND_URL}/search-similar-product`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ product_id: product.id }),
        signal: ctrl.signal,
      })
      const data = await res.json()
      if (addAssistantResponse(data)) return
      setMessages((m) => [
        ...m,
        { role: 'assistant', content: 'Unexpected backend response.' }
      ])
    } catch (e) {
      if (e.name !== 'AbortError') showError('Failed to reuse this product image for search.')
    } finally {
      setLoading(false)
      abortCtrlRef.current = null
    }
  }

  const displayMessages =
    streamingMessage !== null
      ? [
          ...messages.slice(0, messages.length - 1),
          { role: 'assistant', content: streamingMessage }
        ]
      : messages

  return (
    <Box display="flex" flexDirection="column" height="100vh">
      <AppBar position="static" color="default" elevation={1}>
        <Toolbar>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            ShopSense
          </Typography>
          <IconButton onClick={toggleMode} color="inherit">
            {mode === 'light' ? <DarkModeIcon /> : <LightModeIcon />}
          </IconButton>
        </Toolbar>
      </AppBar>

      <Box flex={1} overflow="auto" p={2} bgcolor="background.default">
        <ChatWindow
          messages={displayMessages}
          loading={loading}
          onImageLoad={scrollToBottom}
          onSearchSimilar={handleSearchSimilar}
        />
        <div ref={bottomRef} />
      </Box>

      <Box
        borderTop="1px solid"
        borderColor="divider"
        p={2}
        display="flex"
        justifyContent="center"
      >
        <Box width="100%" maxWidth="800px">
          <ChatInput
            onSend={sendMessage}
            disabled={loading || streamingMessage !== null}
            isGenerating={loading || streamingMessage !== null}
            onStop={handleStop}
          />
        </Box>
      </Box>

      <Snackbar
        open={!!errorMsg}
        autoHideDuration={3000}
        message={errorMsg}
        onClose={() => setErrorMsg('')}
      />
    </Box>
  )
}
